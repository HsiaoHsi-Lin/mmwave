#include <Servo.h>
Servo sv;

const int PIN = 9;
const int CENTER = 1500;      // 中心脈衝
const float F = 8.0;          // 震顫頻率 (Hz)
const int AMP_US = 80;        // ±15 μs → 約 P-P = 2 mm (臂長約 30 mm)
const float DITHER_F = 80.0;  // 微抖動頻率
const int DITHER_US = 10;      // ±8 μs 抖動 (幫助克服死區)

void setup() {
  sv.attach(PIN, 500, 2500);
  sv.writeMicroseconds(CENTER); // 歸中
}

void loop() {
  const float t = micros() / 1e6; // 秒
  float carrier = sinf(2.0f * PI * F * t);
  float dither  = sinf(2.0f * PI * DITHER_F * t);
  int pulse = CENTER + (int)(carrier * AMP_US + dither * DITHER_US);
  sv.writeMicroseconds(pulse);
  delayMicroseconds(1500); // 節流
}

// #include <Servo.h>
// Servo sv;

// const int PIN = 9;

// // 假設 1500 為大概停止點，不需要精準
// const int BASE = 1500;

// // 震顫頻率 (速度反轉頻率)
// const float F = 4.0f;

// // 速度幅度（越大越明顯），建議先從小的開始
// int AMP_US = 500;

// // 微抖動：幫助跨死區（可設 0 禁用）
// const float DITHER_F = 75.0f;
// const int   DITHER_US = 6;

// // 自動偏移補償
// float autoOffset = 0;

// void setup() {
//   sv.attach(PIN, 1000, 2000);
// }

// void loop() {
//   const float t = micros() / 1e6f;
//   float carrier = sinf(2.0f * PI * F * t);          
//   float dither  = sinf(2.0f * PI * DITHER_F * t);  

//   // 核心：如果偏轉太多，offset 自動慢慢修回來
//   autoOffset *= 0.9995f;   

//   int pulse = BASE + (int)(carrier * AMP_US + dither * DITHER_US + autoOffset);
//   pulse = constrain(pulse, 1000, 2000);

//   sv.writeMicroseconds(pulse);
//   delay(8);
// }

// #include <Servo.h>
// Servo sv;

// void setup() {
//   sv.attach(9); // D9
// }

// void loop() {
//   // 三段測試：逆轉 → 停止 → 正轉
//   sv.writeMicroseconds(1400);  // 向一邊轉
//   delay(2000);

//   sv.writeMicroseconds(1500);  // 停止
//   delay(2000);

//   sv.writeMicroseconds(1600);  // 向另一邊轉
//   delay(2000);
// }
