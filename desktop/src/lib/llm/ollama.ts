import { Llm, type LlmConfig } from './index'
import { invoke } from '@tauri-apps/api/core'

export function defaultConfig() {
	return {
		enabled: false,
		model: 'llama3.2:latest',
		ollamaBaseUrl: 'http://localhost:11434',
		platform: 'ollama',
		prompt: `Please summarize the following transcription: \n\n"""\n%s\n"""\n`,

		claudeApiKey: '',
	} satisfies LlmConfig
}

export class Ollama implements Llm {
	private config: LlmConfig

	constructor(config: LlmConfig) {
		this.config = config
	}

	async ask(prompt: string): Promise<string> {
		try {
			// Use Tauri invoke to call the Rust backend
			const response = await invoke<string>('check_ollama_connection', {
				baseUrl: this.config.ollamaBaseUrl,
				model: this.config.model,
				prompt,
			})
			return response
		} catch (error) {
			console.error('Ollama invoke error:', error)
			throw error
		}
	}
}
