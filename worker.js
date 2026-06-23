// ดึงข้อมูลผ่านค่ายหลักเวอร์ชัน 3
import { pipeline, env, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0';

env.allowLocalModels = false;

let generator = null;

self.addEventListener('message', async (event) => {
    const { type, prompt } = event.data;

    if (type === 'load') {
        try {
            // 🌟 กลับมาใช้ Qwen2.5-0.5B ตัวที่ไฟล์ระบบแน่นๆ ครบถ้วนชัวร์ๆ 
            // 🛠️ แต่ใส่คำสั่ง `dtype: 'q4'` เพื่อบังคับโหลดรุ่นประหยัดแรมพิเศษ (ตัวเบาหวิว iPad รันรอด)
            generator = await pipeline('text-generation', 'onnx-community/Qwen2.5-0.5B-Instruct', {
                device: 'wasm', 
                dtype: 'q4', // 👈 ตัวนี้แหละครับที่จะช่วยเซฟแรมไม่ให้หน้าเว็บวูบดับ
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

            // ตั้งค่าให้ตอบสั้น กระชับ แรมนิ่ง
            await generator(prompt, {
                max_new_tokens: 80,
                temperature: 0.4,
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
