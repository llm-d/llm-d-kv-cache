/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef CGO_FUNCTIONS_H
#define CGO_FUNCTIONS_H

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// === FUNCTION DECLARATIONS ===

// Initialize Python interpreter
int Py_InitializeGo();

// Finalize Python interpreter
void Py_FinalizeGo();

// CGo cannot call C macros, so we wrap PyRun_SimpleString in a function
int Go_PyRun_SimpleString(const char* code);

// Wrapper for PyImport_AddModule
PyObject* Go_PyImport_AddModule(const char* name);

// Wrapper for PyModule_GetDict
PyObject* Go_PyModule_GetDict(PyObject* module);

// Wrapper for PyDict_GetItemString
PyObject* Go_PyDict_GetItemString(PyObject* dict, const char* key);

// Helper function to convert Python string to Go string
const char* PyUnicode_AsGoString(PyObject* obj);

// === NEW CACHING FUNCTIONS FOR OPTIMIZATION ===

// Global variables to hold cached module and functions
extern PyObject* g_chat_template_module;
extern PyObject* g_get_or_create_tokenizer_key_func;
extern PyObject* g_apply_chat_template_func;
extern PyObject* g_encode_func;

// Initialize the cached module and functions (call once at startup)
int Py_InitChatTemplateModule();

// Call the cached get_or_create_tokenizer_key function
char* Py_CallGetOrCreateTokenizerKey(const char* json_request);

// Internal function that does the actual work
char* Py_CallGetOrCreateTokenizerKeyInternal(const char* json_request);

// Call the cached apply_chat_template function
char* Py_CallApplyChatTemplate(const char* json_request);

// Internal function that does the actual work
char* Py_CallApplyChatTemplateInternal(const char* json_request);

// Call the cached encode function
char* Py_CallEncode(const char* json_request);

// Internal function that does the actual work
char* Py_CallEncodeInternal(const char* json_request);

// Clear all caches for testing purposes
char* Py_ClearCaches(void);

// Clean up cached objects
void Py_CleanupChatTemplateModule();

// Re-initialize Python interpreter state
int Py_ReinitializeGo();

#endif  // CGO_FUNCTIONS_H