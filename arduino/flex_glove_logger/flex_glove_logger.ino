/*
  Flex-sensor glove logger  —  Arduino MEGA 2560
  ---------------------------------------------------------------------------
  Seven identical voltage-divider channels:
      A0..A4  finger flexion (thumb, index, middle, ring, little)  - sensor along the finger
      A5      index-middle ABDUCTION (spread) - sensor laid across the gap between the
              proximal phalanges of index and middle, on the back of the hand (U vs V, R)
      A6      ELBOW flexion - sensor on the back of the elbow (G, H, P, Q, J, Z)
  Circuit (per channel):
      5V ── [FLEX SENSOR] ──┬── A0..A6
                            │
                          [R_fixed 47k]   (+ optional 100 nF from Ax to GND)
                            │
                           GND
  Optional IMU MPU-6050 (GY-521):  VCC->5V, GND->GND, SDA->pin 20, SCL->pin 21 (MEGA)
                                   (UNO is NOT usable: it has only 6 analog inputs)

  Output: one CSV line every 20 ms (50 Hz):
      t_ms,f0,f1,f2,f3,f4,f5,f6,ax,ay,az,gx,gy,gz
  f0..f6 = 10-bit ADC (0..1023), averaged over 4 conversions.
  IMU fields are raw MPU-6050 counts, or 0 if no IMU is detected.
  Lines starting with '#' are status messages.
*/
#include <Wire.h>

const uint8_t  N_FLEX = 7;
// thumb, index, middle, ring, little, index-middle abduction, elbow
const uint8_t  FLEX_PINS[N_FLEX] = {A0, A1, A2, A3, A4, A5, A6};
const uint32_t PERIOD_MS = 20;        // 50 Hz
const uint8_t  OVERSAMPLE = 4;
const uint8_t  MPU_ADDR = 0x68;

bool imuOK = false;
uint32_t nextT = 0;

int readFlex(uint8_t pin) {
  analogRead(pin);                    // dummy read: lets ADC sample cap settle (high source impedance)
  delayMicroseconds(50);
  uint16_t s = 0;
  for (uint8_t i = 0; i < OVERSAMPLE; i++) s += analogRead(pin);
  return s / OVERSAMPLE;
}

bool imuInit() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0x00);           // wake up
  if (Wire.endTransmission() != 0) return false;
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B); Wire.write(0x00);           // gyro  ±250 dps
  Wire.endTransmission();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C); Wire.write(0x00);           // accel ±2 g
  Wire.endTransmission();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1A); Wire.write(0x03);           // DLPF ~44 Hz
  Wire.endTransmission();
  return true;
}

void imuRead(int16_t *v) {                      // v[0..5] = ax ay az gx gy gz
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom((int)MPU_ADDR, 14);
  int16_t raw[7];
  for (uint8_t i = 0; i < 7; i++) {
    uint8_t hi = Wire.read(), lo = Wire.read();
    raw[i] = (int16_t)((hi << 8) | lo);
  }
  v[0] = raw[0]; v[1] = raw[1]; v[2] = raw[2];  // raw[3] = temperature (skipped)
  v[3] = raw[4]; v[4] = raw[5]; v[5] = raw[6];
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(3000, true);
#endif
  imuOK = imuInit();
  Serial.print(F("# READY imu="));
  Serial.println(imuOK ? 1 : 0);
  nextT = millis();
}

void loop() {
  uint32_t now = millis();
  if ((int32_t)(now - nextT) < 0) return;
  nextT += PERIOD_MS;

  int f[N_FLEX];
  for (uint8_t i = 0; i < N_FLEX; i++) f[i] = readFlex(FLEX_PINS[i]);

  int16_t imu[6] = {0, 0, 0, 0, 0, 0};
  if (imuOK) imuRead(imu);

  Serial.print(now);
  for (uint8_t i = 0; i < N_FLEX; i++) { Serial.print(','); Serial.print(f[i]); }
  for (uint8_t i = 0; i < 6; i++)      { Serial.print(','); Serial.print(imu[i]); }
  Serial.println();
}
