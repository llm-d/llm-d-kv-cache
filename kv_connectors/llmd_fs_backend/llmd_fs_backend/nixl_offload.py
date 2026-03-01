"""Storage Offload Engine for managing asynchronous GPU-Storage transfers."""

import hashlib
import time
from typing import List
import torch
from nixl._api import nixl_agent, nixl_agent_config
from nixl._bindings import nixlBackendError
from nixl.logging import get_logger
from vllm.v1.kv_offload.worker.worker import TransferResult
from collections import deque

def obj_key_to_dev_id(obj_key: str) -> int:
    return int(hashlib.md5(obj_key.encode()).hexdigest(), 16) % (2**31)

class StorageOffloadEngine:
    """
    Engine for managing asynchronous transfers between GPU and storage.
    
    This engine handles:
    - Asynchronous GPU -> Storage (store) operations
    - Asynchronous Storage -> GPU (load) operations
    """
    
    def __init__(
        self,
        io_threads: int,
        gpu_blocks_per_file: int,
        tensors: List[torch.Tensor],
    ):
        """
        Initialize the StorageOffloadEngine with NIXL in OBJ mode.
        
        Args:
            io_threads: Number of I/O threads for parallel transfers
            gpu_blocks_per_file: Number of GPU blocks grouped into a single file
            tensors: List of KV-cache tensors to manage
        """
        self.io_threads = io_threads
        self.gpu_blocks_per_file = gpu_blocks_per_file
        self.tensors = tensors
        self.logger = get_logger(__name__)
        self.backend = "OBJ" # TODO: this should be a param 

        # queue of transfers (job_id, stream, event)
        self._transfers: deque[tuple[int, object, object]] = deque()
        self.logger.info("init len _transfers=%d", len(self._transfers))
        
        # Step 1: Create agent configuration and initialize NIXL agent
        agent_config = nixl_agent_config(backends=[])
        self.agent = nixl_agent("StorageOffloadEngine", agent_config)
        
        # Step 2: Verify OBJ plugin is available
        plugin_list = self.agent.get_plugin_list()
        if "OBJ" not in plugin_list:
            raise RuntimeError("OBJ plugin not available in NIXL")
        
        # Step 3: Log plugin parameters
        self.logger.info(
            "OBJ Plugin parameters:\n%s\n%s",
            self.agent.get_plugin_mem_types("OBJ"),
            self.agent.get_plugin_params("OBJ"),
        )
        
        # Step 4: Create OBJ backend
        self.agent.create_backend("OBJ", {
            "bucket": "testing1",
            "endpoint_override": "http://172.30.228.75:9000",
            "scheme": "http",
            "access_key": "minioadmin",
            "secret_key": "minioadmin",
            #"crtMinLimit": "1000000",
            #"accelerated": "true",
        })
        #self.agent.create_backend("OBJ", {
        #    "bucket": "my-bucket",
        #    "endpoint_override": "http://...",
        #    "scheme": "http",
        #})
        
        # Step 5: Log backend parameters
        self.logger.info(
            "OBJ Backend parameters:\n%s\n%s",
            self.agent.get_backend_mem_types("OBJ"),
            self.agent.get_backend_params("OBJ"),
        )
        
        self.logger.info(
            "Tensors len: %d, ", len(tensors)
        )
        self.logger.info(
            "Tensor[0] shape: %s", tensors[0].shape
        )

        #TODO: no need to reg VRAM for OBJ but will need to do so for other backends
        # self.reg_descs = self.agent.get_reg_descs(self.tensors)
        # assert self.agent.register_memory(self.reg_descs) is not None
        # self._finalizer = weakref.finalize(self, self.agent.deregister_memory, self.reg_descs)

        assert len(tensors) == 1

    def get_blocks_data(self, 
                        tensors: List[torch.Tensor], 
                        block_ids: List[List[int]],
                        bytes_per_block: int
                        ) -> list[tuple[int, int, int]]:
        blocks_data: list[tuple[int, int, int]] = []

        assert len(block_ids) == len(tensors) # TODO: only true for cpu tensors
        for block_list, tensor in zip(block_ids, tensors):
            assert len(block_list) == 1 # TODO: only true for OBJ
            for block in block_list:
                cur_block: list[tuple[int, int, int]] = []
                base_addr = tensor.data_ptr()
                # TODO: for cpu tensors addr is the base addr, since the block is a 1 block slice.
                #       But for gpu tensors we need to do addr = base_addr + (block * bytes_per_block)
                addr = base_addr
                if tensor.is_cuda:
                    device_id = tensor.device.index
                else:
                    device_id = 0
                cur_block = (addr, bytes_per_block, device_id)

                blocks_data.append(cur_block)

        return (blocks_data)

    def async_transfer_gpu_blocks(
        self,
        job_id: int,
        tensors: List[torch.Tensor],
        nixl_reg_dlist,
        files: List[str],
        block_ids: List[List[int]],
        op: str
    ) -> bool:

        self.logger.info("job_id=%d", job_id)
        block_size = self.tensors[0].stride(0) * self.tensors[0].element_size()

        # Open files and set up the xfer_files parameter used by initialize_xfer
        fd_list=[]
        nixl_files=[]
        for file in files:
            #os.makedirs(os.path.dirname(file), exist_ok=True)
            #fd = os.open(file, os.O_RDWR| os.O_CREAT)
            #self.logger.info("file created/opened")
            #assert fd >= 0
            #fd_list.append(fd)
            fd_list.append(file)
        
        # Set up the xfer_desc used by initialize_xfer
        # reg_descs = self.agent.get_reg_descs(self.tensors)
        # assert self.agent.register_memory(reg_descs) is not None

        blocks_data = self.get_blocks_data(tensors, block_ids, block_size) 
        assert blocks_data is not None and len(blocks_data) > 0
        block_num=0
        for cur_block in blocks_data:
            if block_num == 0:
                block_offset = block_ids[0][0] % self.gpu_blocks_per_file
            else:
                block_offset += 1
                block_offset %= self.gpu_blocks_per_file
 
            if self.backend == "OBJ":
                obj_key = fd_list[block_num]
                nixl_files.append((0,
                                   block_size, 
                                   obj_key_to_dev_id(obj_key),
                                   obj_key))
            else:
                missing_blocks = block_ids[0][0] % self.gpu_blocks_per_file
                block_offset = block_num % len(block_ids[0])
                nixl_files.append((block_size*block_offset, 
                                   block_size, 
                                   fd_list[(block_num + missing_blocks) // self.gpu_blocks_per_file], 
                                    "not_obj"))

            block_num += 1
            
        assert len(blocks_data) == len(nixl_files)
        xfer_desc = self.agent.get_xfer_descs(blocks_data, "DRAM") # TODO: VRAM or auto detectd? assumes OBJ.
        assert xfer_desc is not None
            
        files_desc = self.agent.register_memory(nixl_files, "OBJ")
        assert files_desc is not None
        xfer_files = files_desc.trim() # removes the meta data from nixlRegDList
        # self.logger.info("file:%d, desc=%d", layer*block_size*len(block_in_layer)+cur_block*block_size, addr)

        xfer_handle = self.agent.initialize_xfer(
            op, xfer_desc, xfer_files, "StorageOffloadEngine",
        )
        if not xfer_handle:
            self.logger.error("initialize transfer failed.")
            self.agent.deregister_memory(files_desc)
            if nixl_reg_dlist:
                self.agent.deregister_memory(nixl_reg_dlist)
            return False
        state = self.agent.transfer(xfer_handle)
        assert state != "ERR" and state != "NIXL_ERR_BACKEND"
        self._transfers.append((job_id, 
                                xfer_handle, 
                                files_desc, 
                                tensors, 
                                nixl_reg_dlist,
                                block_ids if op == "READ" else None))
    
        #for fd in fd_list:
        #    os.close(fd)

        return True
    
    def async_store_gpu_blocks(
        self,
        job_id: int,
        files: List[str],
        block_ids: List[List[int]]
    ) -> bool:
        """
        Asynchronously store GPU blocks to storage files.
        
        Args:
            job_id: Unique identifier for this transfer job
            files: List of destination file paths
            block_ids: List of block ID lists, one per file
            
        Returns:
            True if the job was successfully submitted
        """
        if self.backend == "OBJ":
            cpu_tensors = []
            for block_id in block_ids:
                assert len(block_id) == 1 # for OBJ we have one block per object
                assert len(self.tensors) == 1 #TODO: assume cross layer
                gpu_slice = self.tensors[0][block_id].contiguous()
                cpu_slice = gpu_slice.cpu()
                cpu_tensors.append(cpu_slice)
            tensors = cpu_tensors
            nixl_reg_dlist = self.agent.register_memory(cpu_tensors)
        else:
            tensors = self.tensors
            nixl_reg_dlist = None

        return self.async_transfer_gpu_blocks(job_id, tensors, nixl_reg_dlist, files, block_ids, "WRITE")

    def async_load_gpu_blocks(
        self,
        job_id: int,
        files: List[str],
        block_ids: List[List[int]]
    ) -> bool:
        """
        Asynchronously load blocks from storage files to GPU.
        
        Args:
            job_id: Unique identifier for this transfer job
            files: List of source file paths
            block_ids: List of block ID lists, one per file
            
        Returns:
            True if the job was successfully submitted
        """
        if self.backend == "OBJ":
            cpu_tensors = []
            for block_id in block_ids:
                assert len(block_id) == 1 # for OBJ we have one block per object
                assert len(self.tensors) == 1 #TODO: assume cross layer
                cpu_slice = torch.empty_like(self.tensors[0][block_id], device='cpu')
                cpu_tensors.append(cpu_slice)
            nixl_reg_dlist = self.agent.register_memory(cpu_tensors)
            tensors = cpu_tensors
        else:
            tensors = self.tensors
            nixl_reg_dlist = None
        return self.async_transfer_gpu_blocks(job_id, tensors, nixl_reg_dlist, files, block_ids, "READ")

    def _complete_transfer(self, entry: tuple):
        """Cleanup resources and copy data back to GPU for a completed transfer."""
        _, xfer_handle, files_desc, cpu_tensors, nixl_reg_dlist, block_ids = entry
        self.agent.deregister_memory(files_desc)
        if nixl_reg_dlist:
            self.agent.deregister_memory(nixl_reg_dlist)
        if block_ids is not None:
            for block_id, cpu_slice in zip(block_ids, cpu_tensors):
                self.tensors[0][block_id].copy_(cpu_slice)
        self.agent.release_xfer_handle(xfer_handle)

    def get_finished(self) -> List[TransferResult]:
        """
        Poll for completed transfer jobs.
        
        Returns:
            List of completed transfer results
        """
        logger = self.logger
        results: list[TransferResult] = []
        to_remove = []
        self.logger.info("get_finished len _transfers=%d", len(self._transfers))
        for entry in self._transfers:
            job_id, xfer_handle = entry[0], entry[1]
            assert job_id is not None and xfer_handle is not None
            try:
                xfer_state = self.agent.check_xfer_state(xfer_handle)
            except nixlBackendError as e:
                logger.error("NIXL backend error for job %s: %s", job_id, e)
                #self._complete_transfer(entry)
                #results.append((job_id, False))
                #to_remove.append(entry)
                continue
            assert xfer_state in ("DONE", "PROC"), f"unexpected xfer_state: {xfer_state}"
            if xfer_state == "DONE":
                self.logger.info("get finished done")
                self._complete_transfer(entry)
                results.append((job_id, True))
                to_remove.append(entry)
            elif xfer_state == "PROC":
                continue
            else:
                logger.error(
                    "NIXL transfer failed for request %s with state "
                    "%s. Marking blocks as invalid.",
                    job_id,
                    xfer_state,
                )
                self._complete_transfer(entry)
                results.append((job_id, False))
                to_remove.append(entry)
        for entry in to_remove:
            self.logger.info("get_finished job_id=%d", job_id)
            self._transfers.remove(entry)
        return results
    
    def wait_job(self, job_id: int):
        """
        Block until the specified job completes.
        
        Args:
            job_id: The job ID to wait for
        """
        self.logger.error("in wait_job() job_id=%d", job_id)
        entry = None
        for e in self._transfers:
            if e[0] == job_id:
                entry = e
                break
        if entry is None:
            # Perhaps the job already completed and processed in get_finished?
            self.logger.error("wait job: job_id not found")
            return
        xfer_handle = entry[1]
        done = False
        i=0
        while not done:
            state = self.agent.check_xfer_state(xfer_handle)
            if state == "DONE":
                self.logger.info("wait finished done")
                done = True
                try:
                    self._transfers.remove(entry)
                    self._complete_transfer(entry)
                except ValueError:
                    pass  # get_finished already processed this entry
            elif state == "PROC":
                i += 1
                if i%10 == 0:
                    self.logger.info("i=%d", i)
                time.sleep(1)
            else:
                self.logger.error("Transfer got error state: %s", state)
                done = True
                try:
                    self._transfers.remove(entry)
                    self._complete_transfer(entry)
                except ValueError:
                    pass  # get_finished already processed this entry
    
    def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        # TODO
        return 
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()
