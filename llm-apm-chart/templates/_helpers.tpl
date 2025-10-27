{{- /* helpers for llm-apm chart — provide both llm-apm.* and llm-apm-chart.*  compatibility */ -}}

{{- define "llm-apm.fullname" -}}
{{- printf "%s" .Release.Name -}}
{{- end -}}

{{- define "llm-apm-chart.fullname" -}}
{{- include "llm-apm.fullname" . -}}
{{- end -}}

{{- define "llm-apm.labels" -}}
app.kubernetes.io/name: {{ include "llm-apm.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: Helm
{{- end -}}

{{- define "llm-apm-chart.labels" -}}
{{- include "llm-apm.labels" . -}}
{{- end -}}
