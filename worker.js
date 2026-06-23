// เรียกใช้งานโครงสร้าง Transformers.js v3 ผ่าน CDN หลัก
import { pipeline, env, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0';

// บังคับให้โหลดออนไลน์จาก Hugging Face เท่านั้น ไม่ใช้โมเดลท้องถิ่น
env.allowLocalModels = false;

let generator = null;

self.addEventListener('message', async (event) => {
    const { type, prompt } = event.data;

    // ส่วนที่ 1: ขั้นตอนการโหลดโมเดลเข้าหน่วยความจำ
    if (type === 'load') {
        try {
            // ใช้ Qwen 0.5B รุ่นที่ไฟล์ครบถ้วนบนเซิร์ฟเวอร์
            // บีบอัดความจำให้อยู่ในโหมด 4-bit (dtype: 'q4') เพื่อป้องกัน iPad แรมน็อค
            generator = await pipeline('text-generation', 'onnx-community/Qwen2.5-0.5B-Instruct', {
                device: 'wasm', 
                dtype: 'q4', 
                progress_callback: (progressData) => {
                    self.postMessage({ action: 'progress', status: progressData.status, data: progressData });
                }
            });
            self.postMessage({ action: 'ready' });
        } catch (error) {
            // 💡 ทริคสำรอง: ถ้า Qwen ยังทำให้ iPad ของคุณรีเซ็ตหน้าจอ ให้ลบเครื่องหมาย // หน้า 3 บรรทัดล่างนี้ออกเพื่อใช้โมเดลขนาดจิ๋วที่สุดในโลกแทน
            // generator = await pipeline('text-generation', 'Xenova/gpt2', { device: 'wasm', progress_callback: (p) => { self.postMessage({ action: 'progress', status: p.status, data: p }); } });
            // self.postMessage({ action: 'ready' }); return;
            
            self.postMessage({ action: 'error', data: error.message });
        }
    }

    // ส่วนที่ 2: ขั้นตอนการคิดและตอบข้อความ
    if (type === 'generate') {
        if (!generator) return;
        
        try {
            // แจ้งหน้าบ้านว่ากำลังเริ่มพ่นคำตอบออกมานะ
            self.postMessage({ action: 'stream_start' });

            // ใช้ TextStreamer ดักจับคำที่ประมวลผลเสร็จแล้วส่งออกไปทันทีโดยไม่ต้องรอเสร็จทั้งประโยค
            const streamer = new TextStreamer(generator.tokenizer, {
                skip_prompt: true,
                callback_function: (tokenText) => {
                    self.postMessage({ action: 'stream', data: tokenText });
                }
            });

            // สั่งคำนวณ โดยจำกัดจำนวนคำไว้ที่ 80 ตัวอักษรเพื่อความปลอดภัยของระบบ
            await generator(prompt, {
                max_new_tokens: 80,
                temperature: 0.5,
                repetition_penalty: 1.15,
                do_sample: true,
                streamer: streamer 
            });
            
            // แจ้งหน้าบ้านว่าคิดคำตอบเสร็จสิ้นแล้ว
            self.postMessage({ action: 'result' });
        } catch (error) {
            self.postMessage({ action: 'error', data: error.message });
        }
    }
});
