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

// Package version exposes build metadata for kv-cache binaries. Values may be
// injected at link time with -ldflags "-X .../version.BuildRef=... -X .../version.CommitSHA=...".
// When not injected, they are populated from the Go build info embedded in the binary.
package version

import "runtime/debug"

var (
	// CommitSHA is the git commit the binary was built from.
	CommitSHA string

	// BuildRef is the build ref (tag or branch) the binary was built from.
	BuildRef string
)

func init() {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return
	}

	if BuildRef == "" {
		BuildRef = info.Main.Version
	}

	if CommitSHA == "" {
		for _, setting := range info.Settings {
			if setting.Key == "vcs.revision" {
				CommitSHA = setting.Value
				break
			}
		}
	}
}
