// ดึงข้อมูลผ่านค่ายหลักเวอร์ชัน 3 ล่าสุด
import { pipeline, env, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0';

env.allowLocalModels = false;

let generator = null;

self.addEventListener('message', async (event) => {
    const { type, prompt } = event.data;

    if (type === 'load') {
        try {
            // 🛠️ แก้ไขวิกฤตหน้าเว็บรีโหลดบน iPad: ล็อกโหมดไปที่ 'wasm' เพื่อบีบการกินแรมให้ต่ำที่สุด ป้องกัน iOS สั่งปิดหน้าเว็บล่ม
            generator = await pipeline('text-generation', 'onnx-community/Qwen2.5-0.5B-Instruct', {
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

            // สร้างสายพานเชื่อมต่ออักษรไหล (Streaming) ไปยังหน้าบ้าน
            const streamer = new TextStreamer(generator.tokenizer, {
                skip_prompt: true,
                callback_function: (tokenText) => {
                    self.postMessage({ action: 'stream', data: tokenText });
                }
            });

            // สั่งประมวลผลคำตอบแบบควบคุมโครงสร้างภาษาไทย
            await generator(prompt, {
                max_new_tokens: 80,
                temperature: 0.6,
                repetition_penalty: 1.15,
                do_sample: true,
                streamer: streamer 
            });
            
            self.postMessage({ action: 'result' });
        } catch (error) {
            self.postMessage({ action: 'error', data: error.message });
        }
    }
});
