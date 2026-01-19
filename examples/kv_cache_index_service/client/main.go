// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/llm-d/llm-d-kv-cache/api/indexerpb"
	"github.com/llm-d/llm-d-kv-cache/examples/testdata"
)

const serverAddr = "localhost:50051"

func main() {
	// Connect to the gRPC server
	conn, err := grpc.NewClient(serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to server: %v", err)
	}
	defer conn.Close()

	client := indexerpb.NewIndexerServiceClient(conn)

	// Test GetPodScores
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := getPodScoresRequest(testdata.Prompt, testdata.ModelName, nil)

	resp, err := client.GetPodScores(ctx, req)
	if err != nil {
		log.Printf("GetPodScores failed: %v", err)
		return
	}

	if len(resp.Scores) == 0 {
		fmt.Println("  No scores returned (KV cache might be empty or no matches found)")
	} else {
		for _, score := range resp.Scores {
			fmt.Printf("Pod: %s, Score: %f\n", score.Pod, score.Score)
		}
	}
}

func getPodScoresRequest(prompt, modelName string, podIdentifiers []string) *indexerpb.GetPodScoresRequest {
	return &indexerpb.GetPodScoresRequest{
		Prompt:         prompt,
		ModelName:      modelName,
		PodIdentifiers: podIdentifiers,
	}
}
