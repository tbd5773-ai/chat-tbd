// ดึงข้อมูลผ่านค่ายหลักเวอร์ชัน 3
import { pipeline, env, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0';

env.allowLocalModels = false;

let generator = null;

self.addEventListener('message', async (event) => {
    const { type, prompt } = event.data;

    if (type === 'load') {
        try {
            // 🛠️ เปลี่ยนมาใช้โมเดล SmolLM2-135M ขนาดไฟล์จิ๋วมากประมาณ 90MB รันบน CPU (WASM) แรมเครื่องไม่เต็มแน่นอน
            generator = await pipeline('text-generation', 'onnx-community/SmolLM2-135M-Instruct', {
                device: 'wasm', 
                quantized: true, 
                progress_callback: (progressData) => {
                    self.postMessage({ action: 'progress', status: progressData.status, data: progressData });
                }
            });
            self.postMessage({ action: 'ready' });
        } catch (error) {
            self.postMessage({ action: 'error', data: error.message });
        }
    }

    if (type === 'generate') {
        if (!generator) return;
        
        try {
            self.postMessage({ action: 'stream_start' });

            const streamer = new TextStreamer(generator.tokenizer, {
                skip_prompt: true,
                callback_function: (tokenText) => {
                    self.postMessage({ action: 'stream', data: tokenText });
                }
            });

            // คุมจำนวนการพ่นคำให้อยู่ในเกณฑ์ปลอดภัย สั้น กระชับ แรมไม่ล่ม
            await generator(prompt, {
                max_new_tokens: 80,
                temperature: 0.5,
                repetition_penalty: 1.2,
                do_sample: true,
                streamer: streamer 
            });
            
            self.postMessage({ action: 'result' });
        } catch (error) {
            self.postMessage({ action: 'error', data: error.message });
        }
    }
});
