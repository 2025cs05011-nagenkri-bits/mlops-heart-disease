{{/* Common name */}}
{{- define "heart-disease-api.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Fully qualified app name */}}
{{- define "heart-disease-api.fullname" -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- printf "%s" $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Common labels */}}
{{- define "heart-disease-api.labels" -}}
app.kubernetes.io/name: {{ include "heart-disease-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app: {{ include "heart-disease-api.name" . }}
{{- end -}}

{{/* Selector labels */}}
{{- define "heart-disease-api.selectorLabels" -}}
app: {{ include "heart-disease-api.name" . }}
app.kubernetes.io/name: {{ include "heart-disease-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
