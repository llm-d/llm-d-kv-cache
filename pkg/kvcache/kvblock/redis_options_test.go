/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0
*/

package kvblock

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestUniversalOptionsFromRedisIndexConfig(t *testing.T) {
	cases := []struct {
		name      string
		cfg       *RedisIndexConfig
		wantAddrs []string
		wantMaster string
		wantDB    int
		wantErr   bool
	}{
		{
			name:      "bare host:port gets redis:// prefix",
			cfg:       &RedisIndexConfig{Address: "127.0.0.1:6379"},
			wantAddrs: []string{"127.0.0.1:6379"},
		},
		{
			name:      "redis:// URL with DB component",
			cfg:       &RedisIndexConfig{Address: "redis://127.0.0.1:6379/3"},
			wantAddrs: []string{"127.0.0.1:6379"},
			wantDB:    3,
		},
		{
			name:      "explicit DB overrides URL DB",
			cfg:       &RedisIndexConfig{Address: "redis://127.0.0.1:6379/3", DB: 7},
			wantAddrs: []string{"127.0.0.1:6379"},
			wantDB:    7,
		},
		{
			name:      "valkey:// rewritten to redis://",
			cfg:       &RedisIndexConfig{Address: "valkey://127.0.0.1:6379"},
			wantAddrs: []string{"127.0.0.1:6379"},
		},
		{
			name: "sentinel: addresses + masterName",
			cfg: &RedisIndexConfig{
				Addresses:  []string{"sent1:26379", "sent2:26379", "sent3:26379"},
				MasterName: "heimdall-redis",
				DB:         1,
			},
			wantAddrs:  []string{"sent1:26379", "sent2:26379", "sent3:26379"},
			wantMaster: "heimdall-redis",
			wantDB:     1,
		},
		{
			name: "cluster: multiple addresses without masterName",
			cfg: &RedisIndexConfig{
				Addresses: []string{"node1:6379", "node2:6379", "node3:6379"},
			},
			wantAddrs: []string{"node1:6379", "node2:6379", "node3:6379"},
		},
		{
			name:    "both address and addresses set is an error",
			cfg:     &RedisIndexConfig{Address: "127.0.0.1:6379", Addresses: []string{"a:1"}},
			wantErr: true,
		},
		{
			name:    "neither address nor addresses set is an error",
			cfg:     &RedisIndexConfig{},
			wantErr: true,
		},
		{
			name:    "malformed address URL is an error",
			cfg:     &RedisIndexConfig{Address: "redis://[::garbage"},
			wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			opts, err := universalOptionsFromRedisIndexConfig(tc.cfg)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.wantAddrs, opts.Addrs)
			require.Equal(t, tc.wantMaster, opts.MasterName)
			require.Equal(t, tc.wantDB, opts.DB)
		})
	}
}
