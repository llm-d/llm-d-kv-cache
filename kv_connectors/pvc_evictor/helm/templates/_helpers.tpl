{{/*
_helpers.tpl - Reusable Template Functions

This file defines helper functions for the Helm chart.
Functions here are used for:
- Generating consistent names and labels across all resources
- Validating required values before deployment
- Computing complex values from multiple inputs

Most values (e.g., .Values.pvc.name, .Values.config.*) are accessed directly in templates.
Only reusable logic and validations belong here.
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "pvc-evictor.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "pvc-evictor.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "pvc-evictor.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "pvc-evictor.labels" -}}
helm.sh/chart: {{ include "pvc-evictor.chart" . }}
{{ include "pvc-evictor.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.labels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "pvc-evictor.selectorLabels" -}}
app.kubernetes.io/name: {{ include "pvc-evictor.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "pvc-evictor.serviceAccountName" -}}
{{- default "default" .Values.serviceAccountName }}
{{- end }}

{{/*
Validate required values
*/}}
{{- define "pvc-evictor.validateValues" -}}
{{- if not .Values.pvc.name }}
{{- fail "pvc.name is required. Please set it in values.yaml or with --set pvc.name=<your-pvc-name>" }}
{{- end }}
{{- if not .Values.securityContext.pod.fsGroup }}
{{- fail "securityContext.pod.fsGroup is required. Please set it in values.yaml or with --set securityContext.pod.fsGroup=<fsgroup>" }}
{{- end }}
{{- if not .Values.securityContext.pod.seLinuxOptions.level }}
{{- fail "securityContext.pod.seLinuxOptions.level is required. Please set it in values.yaml or with --set securityContext.pod.seLinuxOptions.level=<level>" }}
{{- end }}
{{- if not .Values.securityContext.container.runAsUser }}
{{- fail "securityContext.container.runAsUser is required. Please set it in values.yaml or with --set securityContext.container.runAsUser=<user>" }}
{{- end }}
{{- end }}