{{/*
Expand the name of the chart.
*/}}
{{- define "ml-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "ml-platform.fullname" -}}
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
{{- define "ml-platform.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ml-platform.labels" -}}
helm.sh/chart: {{ include "ml-platform.chart" . }}
{{ include "ml-platform.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: ml-platform
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ml-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ml-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "ml-platform.serviceAccountName" -}}
{{- if .Values.security.serviceAccount.create }}
{{- default (include "ml-platform.fullname" .) .Values.security.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.security.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create database URL
*/}}
{{- define "ml-platform.databaseUrl" -}}
{{- if .Values.postgresql.enabled }}
postgresql://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ include "ml-platform.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- else }}
{{ .Values.externalDatabase.url }}
{{- end }}
{{- end }}

{{/*
Create Redis URL
*/}}
{{- define "ml-platform.redisUrl" -}}
{{- if .Values.redis.enabled }}
redis://{{ include "ml-platform.fullname" . }}-redis-master:6379
{{- else }}
{{ .Values.externalRedis.url }}
{{- end }}
{{- end }}

{{/*
Create Kafka bootstrap servers
*/}}
{{- define "ml-platform.kafkaBootstrapServers" -}}
{{- if .Values.kafka.enabled }}
{{ include "ml-platform.fullname" . }}-kafka:9092
{{- else }}
{{ .Values.externalKafka.bootstrapServers }}
{{- end }}
{{- end }}

{{/*
Storage class name
*/}}
{{- define "ml-platform.storageClass" -}}
{{- if .Values.global.storageClass }}
{{- .Values.global.storageClass }}
{{- else if .Values.persistence.storageClass }}
{{- .Values.persistence.storageClass }}
{{- end }}
{{- end }}