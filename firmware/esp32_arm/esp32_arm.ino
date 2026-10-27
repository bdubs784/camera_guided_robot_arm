/*
 * ESP32 arm controller: five hobby servos and a NEMA 17 base stepper on a 40:1 worm.
 *
 * The Pi drives it over a UART cable (or WiFi). Every command is a URL path, the same over HTTP and the cable:
 *   /setall?a=s,e,w,r,g,b&ms=800&prof=1    move several joints (blank = leave alone); prof=1 is a clock-timed
 *                                          minimum-jerk move, so it lasts exactly ms however busy the loop is
 *   /tel  /ver  /stop  /power  /trim  /rest  /relax  /home  /basezero  /speed  /diag
 * On the cable a line ends in ~<n>*<XX> (message number, XOR checksum) and a line without a valid check is refused,
 * because motor noise once turned a corrupted command into "shoulder to 0". CHANGELOG.md says why each piece exists.
 *
 * The page at http://<esp32>/ has sliders, power buttons and game-controller driving (browser Gamepad API).
 *
 * Safety: servos boot limp, commanded to a saved rest pose so switching them on does not fling the arm. The joint
 * limits here are the last line of defence; the Pi checks every move against its own safety map first.
 *
 * Libraries (Library Manager): ESP32Servo, FastAccelStepper. WiFi details go in secrets.h (copy secrets.example.h).
 *
 * Wiring (change the pins below to match):
 *   Servos: shoulder 13, elbow 12, wrist 14, wrist rotate 27, gripper 26
 *   Stepper driver: STEP 33, DIR 32, EN 15 (and driver GND to ESP32 GND)
 *     TB6600: PUL+ 33, PUL- GND, DIR+ 32, DIR- GND, ENA via EN or leave off
 *   Shoulder thermistor: 3.3V -- 10k -- GPIO 35 -- NTC -- GND
 *   Pi UART: ESP32 18 (RX) <- Pi GPIO14, ESP32 19 (TX) -> Pi GPIO15
 *
 * USB serial also takes typed commands: 1..5 <angle> | 6 <angle -180..180> | on/off <1-6> | home | basezero |
 * get | speed <deg/s> | help
 */

#include <ESP32Servo.h>
#include <WiFi.h>
#include <WebServer.h>
#include <FastAccelStepper.h>
#include <ArduinoOTA.h>
#include <Preferences.h>

// ---------------- version, Pi cable ----------------
#define FW_VERSION "4.20"
#define BASE_USE_RMT 1               // 1 = RMT step driver (exact counts), 0 = library default
#define OTA_NAME "arm-esp32"        // shows up as a network port in the Arduino IDE
// OTA_PASSWORD (asked for when uploading over WiFi) is in secrets.h
#define PI_UART_RX 18          // ESP32 pin that listens to the Pi's TX (Pi GPIO14, header pin 8)
#define PI_UART_TX 19          // ESP32 pin that talks to the Pi's RX (Pi GPIO15, header pin 10)
#define PI_UART_BAUD 921600
HardwareSerial PiSerial(2);

// One time-based move of one servo (prof 1 or 2). Position comes from the clock, so it is smooth
// and lasts exactly as long as asked, however busy the loop is.
struct Seg { bool active; uint8_t prof; float p0, p1; uint32_t t0; float T; float f, z; };

// Command arguments, filled from an HTTP request or from a cable line.
struct Args {
  static const int MAXA = 12;
  String k[MAXA], v[MAXA];
  int n = 0;
  bool has(const char* key) const { for (int i = 0; i < n; i++) if (k[i] == key) return true; return false; }
  String get(const char* key) const { for (int i = 0; i < n; i++) if (k[i] == key) return v[i]; return String(""); }
  void add(const String& kk, const String& vv) { if (n < MAXA) { k[n] = kk; v[n] = vv; n++; } }
};
void legacyLine(String line);   // defined at the bottom (old typed serial commands)

// ---------------- WiFi ----------------
#define USE_HOME_WIFI true
#include "secrets.h"     // WiFi names and passwords: copy secrets.example.h to secrets.h and fill it in
const char* AP_SSID       = SECRET_AP_SSID;
const char* AP_PASSWORD   = SECRET_AP_PASSWORD;
const char* HOME_SSID     = SECRET_HOME_SSID;
const char* HOME_PASSWORD = SECRET_HOME_PASSWORD;

// ---------------- Safety ----------------
#define START_ATTACHED false     // servos boot limp
#define STEPPER_START_ENABLED false

// ---------------- Servo config ----------------
const int NUM_SERVOS = 5;
int SERVO_PINS[NUM_SERVOS] = {13, 12, 14, 27, 26};
int LIMITS[NUM_SERVOS][2] = {
  {0, 75},    // 1 shoulder: after the horn was reseated (~+90 slider steps) the safety map stops at 72, and
              // anything above ~75 is an old number that points the arm ~40 deg too high. The Pi clamps its
              // own moves, but only this catches the controller, this page, and the rest-pose move after a
              // power cut. Two servos have burned out on this joint; raise it only with the map rebuilt.
  {0, 160},    // 2 elbow (lower limit opened to 0)
  {0, 170},    // 3 wrist (lower limit opened to 0)
  {0, 112},    // 4 wrist rotate (360-scaled). Max 112: above that the hand gets into itself.
  {40, 160}    // 5 gripper: 40 = wide open (65 mm), 150 = closed
};
int RANGE[NUM_SERVOS] = {180, 180, 180, 360, 180};
int HOME_POS[NUM_SERVOS] = {90, 90, 90, 42, 80};    // wrist rotate home = 42, the level hand everything was calibrated at
const char* JOINT_NAMES[NUM_SERVOS] = {"Shoulder","Elbow","Wrist","Wrist rotate","Gripper"};
const int MIN_US = 500, MAX_US = 2500;

// ---------------- Stepper config ----------------
#define STEP_PIN 33
#define DIR_PIN  32
#define EN_PIN   15
#define EN_ACTIVE_LOW true        // TMC2209: LOW=enabled. TB6600 w/ ENA wired: also usually low.
const float MOTOR_STEPS_REV = 400.0;   // 0.9 deg motor
const float MICROSTEPS = 8.0;          // match driver DIP/jumpers
const float GEAR_RATIO = 40.0;         // worm 40:1
const float STEPS_PER_DEG = MOTOR_STEPS_REV * MICROSTEPS * GEAR_RATIO / 360.0;
const float BASE_MIN = -180, BASE_MAX = 180;
float baseSpeedDps = 20.0;             // base cruise speed deg/s
const float BASE_ACCEL_DPS2 = 20.0;    // base acceleration deg/s^2 (was 4: too sluggish to steer by controller)

Servo servos[NUM_SERVOS];
float currentAngle[NUM_SERVOS], targetAngle[NUM_SERVOS];
bool attached[NUM_SERVOS];
// per-joint speed (deg per 10ms tick; x100 = deg/s). shoulder..gripper
float speedBase[NUM_SERVOS] = {0.8, 1.0, 1.5, 2.0, 20};
float degPerStep[NUM_SERVOS] = {0.8, 1.0, 1.5, 2.0, 20};
// Controller glide: when > 0, this joint moves at this many deg per tick instead of
// degPerStep, so it arrives just as the next controller update is due. 0 = normal speed.
float glideStep[NUM_SERVOS] = {0, 0, 0, 0, 0};
Seg seg[NUM_SERVOS];                    // time-based moves (inactive unless prof=1/2 is sent)
long lastSeq = -1;                      // id of the last move command that carried one
uint32_t lastCmdMicros = 0;             // ESP time when the last move command arrived
float telHz = 0; Stream* telPort = NULL; uint32_t lastTelMs = 0;   // telemetry streaming on a cable
String lineUsb, linePi;                 // cable lines being assembled

FastAccelStepperEngine engine;
const char* baseDrv = "none";             // which step driver the base got
FastAccelStepper *stepper = NULL;
bool stepperEnabled = false, baseHomed = false;

// ---- base jog: the controller sends a SPEED for the base, not a position ----
// One long ramped move toward the travel limit, re-timed as the stick moves, so the
// stepper is never re-aimed 14 times a second and never asked to reverse on the fly.
const float JOG_ACCEL_DPS2 = 40.0;          // base acceleration while driving by controller
const unsigned long JOG_TIMEOUT_MS = 300;   // no controller update for this long = ramp to a stop
float jogDps = 0;                           // speed being asked for (signed, 0 = not jogging)
unsigned long jogLastMs = 0;
int jogDir = 0;                             // direction the jog is running in now (0 = none)
int baseLastDir = 0;                        // direction of the latest base move of any kind
uint32_t jogHz = 0;

WebServer server(80);

// ---------------- helpers ----------------
// ---- per-joint trim: a servo swap leaves the horn a few degrees off. The trim shifts the pulse so the
// same command gives the same physical angle as before; every program keeps using the old numbers.
// Set with /trim?j=0&d=-4.5 (saved in flash, survives restarts). /trim alone shows them.
uint8_t stepCore = 0;                   // CPU core for the step driver; 0 keeps it off the loop's core,
                                        // where a 29.5 ms servo write starved its timing (1 = the old way)
uint8_t stepDrv = 1;                    // 1 = RMT, 0 = MCPWM/PCNT
uint16_t servoWriteMs = 0;              // at most one write per servo per this many ms (0 = every loop)
uint32_t restartAt = 0;                 // restart after answering a core=/drv= switch
float REST_POS[5] = {90, 90, 90, 42, 80}; // rest pose (loaded from flash; default = HOME_POS)
bool wakeOnBoot = false;                // switch the servos on at power-up, holding the rest pose
uint32_t relaxOffAt = 0;                // /relax switches the servos off at this time
Preferences prefs;
float TRIM[NUM_SERVOS] = {0, 0, 0, 0, 0};
const float TRIM_MAX = 30.0;

void loadTrims(){
  prefs.begin("arm", true);
  stepCore = prefs.getInt("stcore", 0) == 0 ? 0 : 1;          // test switches
  stepDrv = prefs.getInt("stdrv", 1) == 0 ? 0 : 1;
  servoWriteMs = (uint16_t)constrain(prefs.getInt("swms", 50), 0, 200);   // 50 by default
  for(int j=0;j<NUM_SERVOS;j++){ String k = String("rest") + j; REST_POS[j] = prefs.getFloat(k.c_str(), (float)HOME_POS[j]); }
  wakeOnBoot = prefs.getInt("wake", 0) == 1;
  for(int j=0;j<NUM_SERVOS;j++){ String k = String("trim") + j; TRIM[j] = prefs.getFloat(k.c_str(), 0.0f); }
  prefs.end();
}
void saveSwitch(const char* key, int v){
  prefs.begin("arm", false); prefs.putInt(key, v); prefs.end();
}


void saveTrim(int j){
  prefs.begin("arm", false);
  String k = String("trim") + j; prefs.putFloat(k.c_str(), TRIM[j]);
  prefs.end();
}

// usable range of a joint in command numbers: its LIMITS, narrowed so command + trim stays inside the servo's range
float limLo(int j){ return max((float)LIMITS[j][0], -TRIM[j]); }
float limHi(int j){ return min((float)LIMITS[j][1], (float)RANGE[j] - TRIM[j]); }

int angleToUs(int j, float a){
  float p = constrain(a + TRIM[j], 0.0f, (float)RANGE[j]);
  return MIN_US + (int)((p/(float)RANGE[j])*(MAX_US-MIN_US));
}
// every servo write goes through here. Skips a write that would not change the pulse, and with
// servoWriteMs > 0 writes each servo at most that often (updateServos writes the latest value later).
int lastServoUs[NUM_SERVOS] = {-1, -1, -1, -1, -1};
uint32_t servoWriteMaxUs = 0, servoWrites = 0;      // longest single servo write, and how many writes
uint32_t lastServoWriteMs[NUM_SERVOS] = {0, 0, 0, 0, 0};
void writeServo(int i){
  if(!attached[i]) return;
  int us = angleToUs(i, currentAngle[i]);
  if(us == lastServoUs[i]) return;
  uint32_t now = millis();
  if(servoWriteMs > 0 && now - lastServoWriteMs[i] < servoWriteMs) return;
  uint32_t t0 = micros();
  servos[i].writeMicroseconds(us);
  uint32_t took = micros() - t0;                    // how long one servo write holds the loop up
  if(took > servoWriteMaxUs) servoWriteMaxUs = took;
  servoWrites++;
  lastServoUs[i] = us; lastServoWriteMs[i] = now;
}


void powerServo(int j, bool on){
  if(j<0||j>=NUM_SERVOS) return;
  seg[j].active = false;
  if(on && !attached[j]){
    servos[j].setPeriodHertz(50);
    servos[j].attach(SERVO_PINS[j], MIN_US, MAX_US);
    targetAngle[j]=currentAngle[j];
    lastServoUs[j] = angleToUs(j,currentAngle[j]); lastServoWriteMs[j] = millis();
    servos[j].writeMicroseconds(angleToUs(j,currentAngle[j]));
    attached[j]=true;
  } else if(!on && attached[j]){ servos[j].detach(); attached[j]=false; }
}

// the driver itself is only powered while the base is moving (the worm holds it otherwise).
bool driverAwake = false;
uint32_t baseIdleSince = 0;
bool driverForceAwake = false;          // test: /diag?awake=1 keeps the driver on while the base stands still

const int TEMP_PIN = 35;                // thermistor on the shoulder servo (ADC1: works with WiFi on)
float shoulderC = NAN;                  // smoothed shoulder temperature, deg C (NAN = no sensor)
uint32_t tempLastMs = 0;
const float NTC_R25 = 10000.0f;          // thermistor ohms at 25 C
const float NTC_B = 3950.0f;            // its B value (the usual one for 10k NTCs)
const float NTC_RFIX = 10000.0f;        // the fixed resistor from 3.3 V to GPIO 35
void readShoulderTemp(){                // NTC divider: 3.3V -- RFIX -- pin -- NTC -- GND
  if(millis() - tempLastMs < 50) return;
  tempLastMs = millis();
  uint32_t mv = 0;
  for(int k = 0; k < 8; k++) mv += analogReadMilliVolts(TEMP_PIN);
  float v = mv / 8.0f;
  if(v < 60.0f || v > 3050.0f){ shoulderC = NAN; return; }            // shorted, or open / not wired (pin pulled high)
  float r = NTC_RFIX * v / (3300.0f - v);
  float c = 1.0f / (1.0f / 298.15f + logf(r / NTC_R25) / NTC_B) - 273.15f;
  if(c < -30.0f || c > 130.0f){ shoulderC = NAN; return; }
  shoulderC = isnan(shoulderC) ? c : shoulderC + 0.025f * (c - shoulderC);   // ~2 s smoothing at 20 reads/s
}
// test input: a jumper from the STEP wire to ECHO_PIN counts the pulses that are really on the wire
const int ECHO_PIN = 34;                // input only, nothing else uses it
volatile uint32_t echoCount = 0;
bool echoOn = false;
void IRAM_ATTR echoIsr(){ echoCount++; }
uint32_t loopLastUs = 0, loopMaxUs = 0; // longest gap between loop() runs
const uint32_t DRIVER_SLEEP_MS = 400;

void driverPower(bool on){
  digitalWrite(EN_PIN, EN_ACTIVE_LOW ? (on?LOW:HIGH) : (on?HIGH:LOW));
  if(on && !driverAwake) delay(2);              // let the driver wake before the first step
  driverAwake = on;
}

void wakeDriver(){ if(stepperEnabled && !driverAwake) driverPower(true); baseIdleSince = millis(); }

void sleepDriverWhenIdle(){
  if(!stepper || !driverAwake || driverForceAwake) return;
  if(stepper->isRunning() || jogDps != 0){ baseIdleSince = millis(); return; }
  if(millis() - baseIdleSince >= DRIVER_SLEEP_MS) driverPower(false);
}

void powerStepper(bool on){
  stepperEnabled = on;
  if(!on){ driverPower(false); jogDps = 0; }    // allowed to move or not; the driver wakes only for moves
}

void setServoTarget(int j, float a){
  if(j<0||j>=NUM_SERVOS) return;
  targetAngle[j] = constrain(a, limLo(j), limHi(j));
  glideStep[j] = 0;   // sliders, serial and Home move at normal speed
  seg[j].active = false;
}

float baseAngle(){ return stepper ? stepper->getCurrentPosition()/STEPS_PER_DEG : 0; }

// ---------------- time-based smooth moves ----------------
float minJerk(float x){ if(x <= 0) return 0; if(x >= 1) return 1; return x*x*x*(10.0f + x*(-15.0f + 6.0f*x)); }

float segRef(const Seg& s, float t){ return s.p0 + (s.p1 - s.p0) * minJerk(t / s.T); }

// Where joint should be now. done = true once the move (and its wobble-cancelling tail) is over.
float segPos(Seg& s, uint32_t now, bool& done){
  float t = (uint32_t)(now - s.t0) / 1e6f;
  if(s.prof == 2 && s.f > 0.2f){
    float zz = constrain(s.z, 0.0f, 0.7f);
    float K = expf(-zz * PI / sqrtf(1.0f - zz*zz));
    float A1 = 1.0f / (1.0f + K), A2 = K / (1.0f + K);
    float half = 0.5f / s.f;              // half a wobble period: the second half of the move cancels the first's swing
    done = t >= s.T + half;
    return A1 * segRef(s, t) + A2 * segRef(s, t - half);
  }
  done = t >= s.T;
  return segRef(s, t);
}

void startSegment(int j, float a, float ms, int prof, float f, float z){
  if(j<0||j>=NUM_SERVOS) return;
  float p1 = constrain(a, limLo(j), limHi(j));
  float p0 = currentAngle[j];
  float maxDps = degPerStep[j] * 100.0f;                   // the Speed setting still caps how fast a joint may go
  float Tmin = maxDps > 0 ? 1.875f * fabs(p1 - p0) / maxDps : 0;   // a smooth move peaks at 1.875x its average speed
  float T = max(ms / 1000.0f, Tmin);
  if(T < 0.02f) T = 0.02f;
  seg[j].p0 = p0; seg[j].p1 = p1; seg[j].t0 = micros(); seg[j].T = T;
  seg[j].prof = (uint8_t)prof; seg[j].f = f; seg[j].z = z;
  seg[j].active = true;
  targetAngle[j] = p1; glideStep[j] = 0;
}
// connect the base stepper with the chosen driver (RMT or MCPWM/PCNT)
volatile bool stepperReady = false;
void connectStepper(){
  if(stepCore == 0) engine.init(0); else engine.init();
#if BASE_USE_RMT && defined(DRIVER_RMT)
  if(stepDrv == 1){ stepper = engine.stepperConnectToPin(STEP_PIN, DRIVER_RMT); if(stepper) baseDrv = "rmt"; }
#endif
#if defined(DRIVER_MCPWM_PCNT)
  if(!stepper && stepDrv == 0){ stepper = engine.stepperConnectToPin(STEP_PIN, DRIVER_MCPWM_PCNT); if(stepper) baseDrv = "pcnt"; }
#endif
  if(!stepper){ stepper = engine.stepperConnectToPin(STEP_PIN); baseDrv = "default"; }
  if(stepper){ stepper->setDirectionPin(DIR_PIN); }
  stepperReady = true;
}
void stepperSetupTask(void*){ connectStepper(); vTaskDelete(NULL); }


void baseGoTo(float deg){ baseGoToAt(deg, -1, -1); }

void baseGoToAt(float deg, float dps, float dps2){
  if(!stepper || !stepperEnabled) return;
  deg = constrain(deg, BASE_MIN, BASE_MAX);
  jogDps = 0; jogDir = 0;                       // a slider or script move takes over from the controller
  baseLastDir = (deg > baseAngle()) ? 1 : -1;
  wakeDriver();
  float sp = dps > 0 ? constrain(dps, 0.2f, 40.0f) : baseSpeedDps;            // per-move speed (bs=)
  float ac = dps2 > 0 ? constrain(dps2, 1.0f, 60.0f) : BASE_ACCEL_DPS2;       // and acceleration (ba=)
  stepper->setSpeedInHz((uint32_t)max(1.0f, sp*STEPS_PER_DEG));
  stepper->setAcceleration((uint32_t)max(1.0f, ac*STEPS_PER_DEG));
  stepper->moveTo((int32_t)(deg*STEPS_PER_DEG));
}

void baseJog(float dps){
  jogDps = constrain(dps, -baseSpeedDps, baseSpeedDps);
  jogLastMs = millis();
}

void updateBaseJog(){
  if(!stepper) return;
  if(jogDps != 0 && (!stepperEnabled || millis() - jogLastMs > JOG_TIMEOUT_MS)) jogDps = 0;
  int want = jogDps > 0 ? 1 : (jogDps < 0 ? -1 : 0);
  if(jogDir != 0 && want != jogDir){ stepper->stopMove(); jogDir = 0; }   // released or reversed: ramp down
  if(want == 0) return;
  if(jogDir == 0 && stepper->isRunning() && baseLastDir != want) return;   // wait for standstill before reversing
  uint32_t hz = (uint32_t)(fabs(jogDps) * STEPS_PER_DEG);
  if(hz < 20) hz = 20;
  if(jogDir == 0){
    stepper->setSpeedInHz(hz);
    stepper->setAcceleration((uint32_t)(JOG_ACCEL_DPS2 * STEPS_PER_DEG));
    wakeDriver();
    stepper->moveTo((int32_t)((want > 0 ? BASE_MAX : BASE_MIN) * STEPS_PER_DEG));
    jogDir = want; baseLastDir = want; jogHz = hz;
  } else if(hz != jogHz){
    stepper->setSpeedInHz(hz);
    stepper->applySpeedAcceleration();
    jogHz = hz;
  }
}

// ---- base memory: the worm gear cannot be turned by the arm's weight, so with the power off the base
// stays where it was. Its step count is saved once the base has stood still for a second (at most one flash write
// per move) and restored at power-up. If the power dies mid-move the saved count is off by that move, which
// basehome's search covers.
int32_t baseSaved = INT32_MIN;
uint32_t baseStillSince = 0;
bool baseRestored = false;

void saveBasePos(int32_t p){
  prefs.begin("arm", false); prefs.putInt("basepos", p); prefs.end();
  baseSaved = p;
}

void restoreBasePos(){
  if(!stepper) return;
  prefs.begin("arm", true);
  bool has = prefs.isKey("basepos");
  int32_t p = prefs.getInt("basepos", 0);
  prefs.end();
  if(has){ stepper->setCurrentPosition(p); baseSaved = p; baseRestored = true; baseHomed = true; }
}

void rememberBase(){
  if(!stepper) return;
  uint32_t now = millis();
  int32_t p = stepper->getCurrentPosition();
  if(stepper->isRunning() || p == baseSaved){ baseStillSince = now; return; }
  if(now - baseStillSince >= 1000){ saveBasePos(p); baseStillSince = now; }
}

bool baseSetZero(){
  if(!stepper) return false;
  jogDps = 0; jogDir = 0;
  stepper->forceStopAndNewPosition(0);   // wherever we are IS 0 now
  baseHomed = true;
  saveBasePos(0);
  return true;
}

// ---------------- web page ----------------
const char PAGE[] PROGMEM = R"HTML(
<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arm v4</title><style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;max-width:760px;margin:0 auto;padding:16px}
h2{text-align:center;margin:8px 0 16px}
.joint{background:#1d1d1f;border-radius:12px;padding:12px 14px;margin-bottom:10px}
.row{display:flex;align-items:center;gap:8px}
.name{font-weight:600}.val{margin-left:auto;font-variant-numeric:tabular-nums;color:#7fd}
input[type=range]{width:100%;margin-top:8px;accent-color:#2bd}
input[type=range]:disabled{accent-color:#555}
.btns{display:flex;gap:8px;margin-top:14px}
button{padding:12px;border:0;border-radius:10px;font-size:15px;background:#2a2a2e;color:#eee}
.btns button{flex:1}button:active{background:#3a3a40}
.pw{min-width:58px;padding:6px 0;font-size:13px;font-weight:700;border-radius:8px}
.pw.on{background:#164;color:#7fd}.pw.off{background:#411;color:#f88}
.mini{min-width:52px;padding:6px 4px;font-size:12px;border-radius:8px;background:#233}
#stat{text-align:center;color:#888;font-size:13px;margin-top:10px}
.dim input[type=range]{opacity:.4}
.cams{display:flex;gap:8px;margin-bottom:10px}.cams div{flex:1;min-width:0}
.cams img{width:100%;border-radius:10px;display:block;background:#000;min-height:60px}
.cams span{font-size:12px;color:#888}
.stop{background:#822;font-weight:700}
#padhelp{font-size:12px;color:#888;margin-top:6px;line-height:1.5}
</style></head><body>
<h2>Robot Arm v4</h2>
<div class="cams">
 <div><img id="camT" alt="torre cam offline"><span>Torre</span></div>
 <div><img id="camG" alt="gripper cam offline"><span>Gripper</span></div>
</div>
<div class="joint"><div class="row"><span class="name">Controller</span>
<span class="val" id="padst">not connected</span></div>
<div id="padhelp">Plug in a game controller and press any button. START arms it, B stops everything.<br>
Left stick: base / shoulder &middot; Right stick: wrist rotate / elbow &middot; D-pad up/down: wrist<br>
LT / RT: open / close gripper &middot; hold RB: slow and precise</div></div>
<div id="joints"></div>
<div class="joint"><div class="row"><span class="name">Speed</span>
<span class="val" id="spv">50&deg;/s</span></div>
<input type="range" id="spd" min="5" max="2000" value="100"
 oninput="spv.innerHTML=this.value+'&deg;/s'" onchange="api('/speed?v='+this.value)"></div>
<div class="btns"><button class="stop" onclick="stopAll()">STOP</button><button onclick="goHome()">Servos Home</button>
<button onclick="powerAll(0)">All off</button><button onclick="powerAll(1)">All on</button><button onclick="relax()">Relax</button></div>
<div id="stat">connecting...</div>
<script>
let J=[],busy=false,pend={};
// Pi camera streams (dual_cam_stream.py on the Pi). Change this if the Pi's name/IP differs.
const CAM='http://raspberrypi.local:8080';
camT.src=CAM+'/torre'; camG.src=CAM+'/gripper';
async function api(p){try{const r=await fetch(p);return await r.json();}
 catch(e){stat.textContent='connection lost';return null;}}
function paint(i,on){const b=document.getElementById('p'+i),s=document.getElementById('s'+i),
 c=document.getElementById('card'+i);b.textContent=on?'ON':'OFF';
 b.className='pw '+(on?'on':'off');s.disabled=!on;c.classList.toggle('dim',!on);if(J[i])J[i].on=on?1:0;}
function build(info){J=info.joints;joints.innerHTML='';
 tgt=J.map(j=>j.target);act=J.map(j=>j.angle);sent=tgt.slice();
 J.forEach((j,i)=>{const extra=j.stepper?`<button class="mini" onclick="api('/basezero').then(refresh)">SET ZERO</button>`:'';
  joints.insertAdjacentHTML('beforeend',
  `<div class="joint" id="card${i}"><div class="row">
   <button class="pw" id="p${i}" onclick="togglePower(${i})"></button>
   <span class="name">${j.name}</span>${extra}
   <span class="val" id="v${i}">${j.target}&deg;</span></div>
   <input type="range" id="s${i}" min="${j.min}" max="${j.max}" value="${j.target}"
    oninput="document.getElementById('v${i}').innerHTML=this.value+'&deg;'"
    onchange="setJoint(${i},this.value)"></div>`);
  paint(i,j.on==1);});
 stat.textContent='connected';}
async function refresh(){const d=await api('/state');if(d)build(d);}
async function togglePower(i){const on=document.getElementById('p'+i).textContent!=='ON';
 const d=await api(`/power?j=${i}&on=${on?1:0}`);if(d)paint(i,on);}
async function powerAll(on){const d=await api('/power?on='+on);
 if(d)J.forEach((j,i)=>paint(i,on==1));}
async function setJoint(i,a){tgt[i]=act[i]=sent[i]=+a;if(busy){pend[i]=a;return;}busy=true;
 await api(`/set?j=${i}&a=${a}`);busy=false;
 if(i in pend){const v=pend[i];delete pend[i];setJoint(i,v);}}
async function relax(){const d=await api('/relax');if(!d)return;
 d.joints.forEach((j,i)=>{if(j.stepper)return;tgt[i]=sent[i]=j.target;
  document.getElementById('s'+i).value=j.target;
  document.getElementById('v'+i).innerHTML=j.target+'&deg;';});
 setTimeout(()=>J.forEach((j,i)=>{if(i<5)paint(i,false);}),3800);}
async function goHome(){const d=await api('/home');if(!d)return;
 d.joints.forEach((j,i)=>{if(j.stepper)return;tgt[i]=sent[i]=j.target;
  document.getElementById('s'+i).value=j.target;
  document.getElementById('v'+i).innerHTML=j.target+'&deg;';});}

// ---------------- game controller ----------------
const PAD_DIR=[1,1,1,1,1,1];        // change a 1 to -1 if that joint moves the wrong way
const PAD_RATE=[30,40,50,60,60,15]; // deg/s at full stick: shoulder, elbow, wrist, wrist rotate, gripper, base
const PAD_LIMIT={3:[0,160]};        // tighter limits on the controller (wrist rotate: protects the camera ribbon)
const LEASH=[10,10,10,15,25,6];     // a target may run at most this many degrees ahead of the real joint (servos only)
const DEAD=0.15;
let tgt=[],act=[],sent=[],armed=false,padBusy=false,lastT=0,lastSend=0,prevBtn=[];
let jog=0,lastJog=0,jogShowUntil=0;   // base is driven by speed (deg/s), not by position
function padStatus(t,c){padst.textContent=t;padst.style.color=c||'#7fd';}
function curPad(){try{const g=navigator.getGamepads?navigator.getGamepads():[];
 for(const p of g)if(p&&p.connected)return p;}catch(e){}return null;}
function dz(v){const a=Math.abs(v||0);if(a<DEAD)return 0;const s=(a-DEAD)/(1-DEAD);return Math.sign(v)*s*s;}
function held(p,i){return !!(p.buttons[i]&&p.buttons[i].pressed);}
function edge(p,i){const now=held(p,i),was=prevBtn[i];prevBtn[i]=now;return now&&!was;}
async function stopAll(){armed=false;jog=lastJog=0;await api('/stop');await refresh();
 padStatus(curPad()?'stopped - press START to arm':'not connected','#f88');}
async function armPad(){await refresh();armed=true;padStatus('ARMED - B or STOP to stop','#fa4');}
function disarm(msg){armed=false;padStatus(msg);if(jog||lastJog){jog=lastJog=0;api('/setall?a=&bv=0');}}
async function sendPad(now){
 if(padBusy)return;
 const parts=tgt.map((v,i)=>!J[i].stepper&&Math.abs(v-sent[i])>=0.2?v.toFixed(1):'');
 const bj=jog,jogging=bj!==0||lastJog!==0;   // keep sending while the stick is held, plus one 0 to stop
 const moving=jogging||parts.some(x=>x);
 if(jogging)jogShowUntil=now+1500;
 if(now-lastSend<70)return;                // at most about 14 commands a second, so the ESP32 is not swamped
 if(!moving&&now-lastSend<400)return;      // idle: just ask where the joints are now and then
 padBusy=true;lastSend=now;const snap=tgt.slice();
 const d=await api('/setall?a='+parts.join(',')+(moving?'&ms=150':'')+(jogging?'&bv='+bj.toFixed(1):''));
 parts.forEach((x,i)=>{if(x)sent[i]=snap[i];});
 if(d){lastJog=bj;if(d.a)act=d.a;}
 if(d&&d.a&&now<jogShowUntil)J.forEach((j,i)=>{if(!j.stepper)return;   // show where the base really is
  tgt[i]=sent[i]=act[i];const sl=document.getElementById('s'+i);
  if(sl){sl.value=Math.round(act[i]);document.getElementById('v'+i).innerHTML=Math.round(act[i])+'&deg;';}});
 padBusy=false;
}
function padLoop(t){
 requestAnimationFrame(padLoop);
 const dt=Math.min(0.1,(t-lastT)/1000||0);lastT=t;
 const p=curPad();
 if(!p){if(armed)disarm('controller lost');return;}
 if(edge(p,1)){stopAll();return;}
 if(edge(p,9)){if(armed)disarm('connected - press START to arm');else armPad();}
 if(!armed||!J.length)return;
 const slow=held(p,5)?0.3:1;jog=0;
 const cmd=[-dz(p.axes[1]),-dz(p.axes[3]),(held(p,12)?1:0)-(held(p,13)?1:0),dz(p.axes[2]),
  (p.buttons[7]?p.buttons[7].value:0)-(p.buttons[6]?p.buttons[6].value:0),dz(p.axes[0])];
 for(let i=0;i<J.length&&i<6;i++){
  const c=cmd[i]*PAD_DIR[i];
  if(!c||!J[i].on)continue;
  if(J[i].stepper){jog=c*PAD_RATE[i]*slow;continue;}   // base: send a speed, the ESP32 does the ramp
  let lo=J[i].min,hi=J[i].max;
  if(PAD_LIMIT[i]){lo=Math.max(lo,PAD_LIMIT[i][0]);hi=Math.min(hi,PAD_LIMIT[i][1]);}
  let v=tgt[i]+c*PAD_RATE[i]*slow*dt;
  // leash: never run far ahead of the real joint, and never pull the target backwards
  v=c>0?Math.max(tgt[i],Math.min(v,act[i]+LEASH[i])):Math.min(tgt[i],Math.max(v,act[i]-LEASH[i]));
  v=Math.max(lo,Math.min(hi,v));
  if(c>0&&tgt[i]>hi||c<0&&tgt[i]<lo)v=tgt[i];   // already outside the controller limit: don't jump
  tgt[i]=v;
  const sl=document.getElementById('s'+i);
  if(sl){sl.value=Math.round(v);document.getElementById('v'+i).innerHTML=Math.round(v)+'&deg;';}
 }
 sendPad(t);
}
addEventListener('gamepadconnected',e=>{padStatus('connected - press START to arm');
 if(e.gamepad&&e.gamepad.mapping!=='standard')padhelp.insertAdjacentHTML('beforeend',
  '<br><b>This controller reports a non-standard layout, so buttons may be in different places.</b>');});
addEventListener('gamepaddisconnected',()=>disarm('not connected'));
if(!navigator.getGamepads)padStatus('this browser blocks controllers on this page','#f88');
refresh();requestAnimationFrame(padLoop);
</script></body></html>
)HTML";

// ---------------- state json ----------------
String stateJson(){
  String s="{\"joints\":[";
  for(int i=0;i<NUM_SERVOS;i++){
    if(i) s+=",";
    s+="{\"name\":\""+String(JOINT_NAMES[i])+"\",\"min\":"+String((int)ceil(limLo(i)))
      +",\"max\":"+String((int)floor(limHi(i)))+",\"angle\":"+String((int)currentAngle[i])
      +",\"target\":"+String((int)targetAngle[i])+",\"on\":"+String(attached[i]?1:0)+"}";
  }
  s+=",{\"name\":\"Base (worm)\",\"stepper\":1,\"min\":-180,\"max\":180,\"angle\":"
    +String((int)baseAngle())+",\"target\":"+String((int)baseAngle())
    +",\"on\":"+String(stepperEnabled?1:0)+"}";
  s+="]}";
  return s;
}

// where every joint is right now: {"ok":1,"a":[shoulder,elbow,wrist,wrist rotate,gripper,base]}
String posJson(){
  String s="{\"ok\":1,\"a\":[";
  for(int i=0;i<NUM_SERVOS;i++){ s+=String(currentAngle[i],1); s+=","; }
  s+=String(baseAngle(),1); s+="]}";
  return s;
}

// ---------------- one command handler for HTTP and cable ----------------

String posJsonId(long id){
  String s = posJson();
  s.remove(s.length() - 1);                                  // drop the closing brace
  s += ",\"t\":" + String((unsigned long)micros());
  if(id >= 0) s += ",\"id\":" + String(id);
  s += "}";
  return s;
}

String telJson(){
  String s = "{\"ok\":1,\"tel\":1,\"t\":" + String((unsigned long)micros()) + ",\"ms\":" + String((unsigned long)millis());
  s += ",\"seq\":" + String(lastSeq) + ",\"rx\":" + String((unsigned long)lastCmdMicros) + ",\"p\":[";
  for(int i=0;i<NUM_SERVOS;i++){ if(i) s += ","; s += String(currentAngle[i], 2); }
  s += "],\"tg\":[";
  for(int i=0;i<NUM_SERVOS;i++){ if(i) s += ","; s += String(targetAngle[i], 2); }
  s += "],\"mv\":[";
  for(int i=0;i<NUM_SERVOS;i++){
    if(i) s += ",";
    bool mv = attached[i] && (seg[i].active || fabs(targetAngle[i] - currentAngle[i]) >= 0.01f);
    s += mv ? "1" : "0";
  }
  s += "],\"on\":[";
  for(int i=0;i<NUM_SERVOS;i++){ s += attached[i] ? "1," : "0,"; }
  s += stepperEnabled ? "1" : "0";
  s += "],\"st\":" + (isnan(shoulderC) ? String("null") : String(shoulderC, 1));
  s += ",\"b\":" + String(baseAngle(), 3);
  s += ",\"bt\":" + String(stepper ? stepper->targetPos() / STEPS_PER_DEG : 0.0f, 3);
  s += ",\"br\":" + String((stepper && stepper->isRunning()) ? 1 : 0);
  s += ",\"bv\":" + String(stepper ? stepper->getCurrentSpeedInMilliHz() / 1000.0f / STEPS_PER_DEG : 0.0f, 3);
  s += ",\"lp\":" + String(loopMaxUs / 1000.0f, 1) + ",\"ec\":" + String((unsigned long)echoCount);
  s += ",\"da\":" + String(driverAwake ? 1 : 0);
  s += ",\"wu\":" + String((unsigned long)servoWriteMaxUs) + ",\"wn\":" + String((unsigned long)servoWrites) + "}";
  loopMaxUs = 0; servoWriteMaxUs = 0;
  return s;
}

Args argsFromServer(){
  Args a;
  for(int i=0;i<server.args();i++) a.add(server.argName(i), server.arg(i));
  return a;
}

// true only for a plain number ("12", "-3.5", "+0.25"). String.toFloat() reads junk like "~123" as 0.
bool isNum(const String& v){
  int n = v.length(), i = 0; bool dig = false, dot = false;
  if(n == 0) return false;
  if(v[0] == '-' || v[0] == '+') i = 1;
  for(; i < n; i++){
    char c = v[i];
    if(c >= '0' && c <= '9') dig = true;
    else if(c == '.' && !dot) dot = true;
    else return false;
  }
  return dig;
}

// every listed argument that is present must be a plain number; every non-blank entry of a comma list too.
bool numsOk(const Args& a, const char* const* keys, int nk, const char* listKey){
  for(int k = 0; k < nk; k++) if(a.has(keys[k]) && !isNum(a.get(keys[k]))) return false;
  if(listKey && a.has(listKey)){
    String s = a.get(listKey); int start = 0;
    for(int k = 0; k <= (int)s.length(); k++){
      if(k == (int)s.length() || s[k] == ','){
        if(k > start && !isNum(s.substring(start, k))) return false;
        start = k + 1;
      }
    }
  }
  return true;
}
const char* BAD_NUMBER = "{\"ok\":0,\"error\":\"bad number\"}";

Args argsFromQuery(const String& q){
  Args a; int s = 0, n = q.length();
  while(s < n){
    int e = q.indexOf('&', s); if(e < 0) e = n;
    String part = q.substring(s, e);
    int eq = part.indexOf('=');
    if(eq >= 0) a.add(part.substring(0, eq), part.substring(eq + 1));
    else if(part.length()) a.add(part, "");
    s = e + 1;
  }
  return a;
}

void noteMove(const Args& a){
  lastCmdMicros = micros();
  if(a.has("id")) lastSeq = a.get("id").toInt();
}

String handleCmd(const String& path, const Args& a){
  long id = a.has("id") ? a.get("id").toInt() : -1;
  if(path == "/state") return stateJson();
  if(path == "/tel") return telJson();
  if(path == "/diag"){                                    // test tools
    if(a.has("awake")){
      driverForceAwake = a.get("awake").toInt() == 1;
      if(driverForceAwake && stepperEnabled) driverPower(true);
      baseIdleSince = millis();                           // awake=0: the normal sleep takes over 0.4 s later
    }
    if(a.has("echo")){
      bool on = a.get("echo").toInt() == 1;
      if(on && !echoOn){ pinMode(ECHO_PIN, INPUT); attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echoIsr, RISING); echoOn = true; }
      else if(!on && echoOn){ detachInterrupt(digitalPinToInterrupt(ECHO_PIN)); echoOn = false; }
    }
    if(a.has("zero")) echoCount = 0;
    if(a.has("sw")){ servoWriteMs = (uint16_t)constrain(a.get("sw").toInt(), 0, 200); saveSwitch("swms", servoWriteMs); }
    if(a.has("core")){ saveSwitch("stcore", a.get("core").toInt() == 0 ? 0 : 1); restartAt = millis() + 500; }
    if(a.has("drv")){ saveSwitch("stdrv", a.get("drv") == "pcnt" ? 0 : 1); restartAt = millis() + 500; }
    if(restartAt){ for(int i=0;i<NUM_SERVOS;i++){ targetAngle[i]=currentAngle[i]; seg[i].active=false; } if(stepper) stepper->stopMove(); }
    return String("{\"ok\":1,\"restart\":") + (restartAt ? 1 : 0) + ",\"sw\":" + String(servoWriteMs) + ",\"awake\":" + (driverForceAwake ? 1 : 0) + ",\"da\":" + (driverAwake ? 1 : 0)
           + ",\"echo\":" + (echoOn ? 1 : 0) + ",\"ec\":" + String((unsigned long)echoCount) + "}";
  }
  if(path == "/ver"){
    return String("{\"ok\":1,\"fw\":\"") + FW_VERSION + "\",\"prof\":[0,1,2],\"uart\":[" + String(PI_UART_RX) + ","
           + String(PI_UART_TX) + "],\"baud\":" + String(PI_UART_BAUD) + ",\"bs\":1,\"diag\":1,\"ck\":1,\"st\":1,\"core\":" + String(stepCore) + ",\"sw\":" + String(servoWriteMs) + ",\"drv\":\"" + String(baseDrv) + "\",\"base_restored\":" + String(baseRestored ? 1 : 0)
           + ",\"base\":" + String(baseAngle(), 2) + "}";
  }
  if(path == "/set"){
    { const char* k[] = {"j", "a"}; if(!numsOk(a, k, 2, NULL) || !a.has("j") || !a.has("a")) return BAD_NUMBER; }
    noteMove(a);
    int j = a.get("j").toInt(); int ang = a.get("a").toInt();
    if(j == NUM_SERVOS) baseGoTo(ang); else setServoTarget(j, ang);
    return "{\"ok\":1}";
  }
  if(path == "/setall"){
    // a = comma list: shoulder,elbow,wrist,wrist rotate,gripper,base. Blank entry = leave alone.
    // ms = how long the move should take (the controller sends this so motion is continuous).
    // bv = base speed in deg/s, signed. The controller sends this instead of a base position.
    // prof = 1 or 2 (optional): time-based smooth move; 2 also cancels wobble (needs f and z).
    { const char* k[] = {"ms", "prof", "f", "z", "bs", "ba", "bv"}; if(!numsOk(a, k, 7, "a")) return BAD_NUMBER; }
    noteMove(a);
    String s = a.get("a"); int n = s.length(), idx = 0, start = 0;
    float ms = a.has("ms") ? a.get("ms").toFloat() : 0;
    int prof = a.has("prof") ? a.get("prof").toInt() : 0;
    float f = a.has("f") ? a.get("f").toFloat() : 0;
    float z = a.has("z") ? a.get("z").toFloat() : 0;
    float bs = a.has("bs") ? a.get("bs").toFloat() : -1;     // base speed for this move, deg/s
    float ba = a.has("ba") ? a.get("ba").toFloat() : -1;     // base acceleration, deg/s^2
    for(int k=0;k<=n && idx<=NUM_SERVOS;k++){
      if(k==n || s[k]==','){
        if(k>start){
          float v = s.substring(start,k).toFloat();
          if(idx<NUM_SERVOS){
            if(prof >= 1) startSegment(idx, v, ms, prof, f, z);
            else {
              setServoTarget(idx,v);
              if(ms>0){
                float st=fabs(targetAngle[idx]-currentAngle[idx])/(ms/10.0f);
                glideStep[idx]=constrain(st,0.01f,degPerStep[idx]);
              }
            }
          } else baseGoToAt(v, bs, ba);
        }
        idx++; start=k+1;
      }
    }
    if(a.has("bv")) baseJog(a.get("bv").toFloat());   // bv = base speed in deg/s (controller)
    return posJsonId(id);
  }
  if(path == "/stop"){
    noteMove(a);
    for(int i=0;i<NUM_SERVOS;i++){ targetAngle[i]=currentAngle[i]; glideStep[i]=0; seg[i].active=false; }   // hold where it is
    jogDps = 0; jogDir = 0;
    if(stepper) stepper->stopMove();                                  // base ramps to a stop
    return posJsonId(id);
  }
  if(path == "/speed"){
    float v=constrain(a.get("v").toFloat(),1.0f,2000.0f);
    for(int i=0;i<NUM_SERVOS;i++) degPerStep[i]=speedBase[i]*(v/100.0);
    return "{\"ok\":1}";
  }
  if(path == "/rest"){                                   // rest pose
    bool save = false;
    if(a.has("set")){ for(int j=0;j<NUM_SERVOS;j++) REST_POS[j] = currentAngle[j]; save = true; }
    if(a.has("a")){
      if(!numsOk(a, NULL, 0, "a")) return BAD_NUMBER;
      String s = a.get("a"); int j = 0, start = 0;
      for(int k=0;k<=(int)s.length() && j<NUM_SERVOS;k++){
        if(k==(int)s.length() || s[k]==','){
          if(k>start) REST_POS[j] = constrain(s.substring(start,k).toFloat(), limLo(j), limHi(j));
          j++; start=k+1;
        }
      }
      save = true;
    }
    if(save){
      prefs.begin("arm", false);
      for(int j=0;j<NUM_SERVOS;j++){ String k = String("rest") + j; prefs.putFloat(k.c_str(), REST_POS[j]); }
      prefs.end();
    }
    if(a.has("wake")){ wakeOnBoot = a.get("wake").toInt() == 1; saveSwitch("wake", wakeOnBoot ? 1 : 0); }
    String r = "{\"ok\":1,\"rest\":[";
    for(int j=0;j<NUM_SERVOS;j++){ if(j) r += ","; r += String(REST_POS[j], 1); }
    return r + "],\"wake\":" + String(wakeOnBoot ? 1 : 0) + "}";
  }
  if(path == "/relax"){                                  // glide to the rest pose, then servos off
    noteMove(a);
    float ms = a.has("ms") ? constrain(a.get("ms").toFloat(), 500.0f, 20000.0f) : 3000.0f;
    for(int j=0;j<NUM_SERVOS;j++) if(attached[j]) startSegment(j, REST_POS[j], ms, 1, 0, 0);
    relaxOffAt = millis() + (uint32_t)ms + 600;
    return stateJson();
  }
  if(path == "/home"){
    noteMove(a);
    for(int i=0;i<NUM_SERVOS;i++){ targetAngle[i]=HOME_POS[i]; glideStep[i]=0; seg[i].active=false; }
    return stateJson();
  }
  if(path == "/basezero"){
    bool ok=baseSetZero();
    return ok ? "{\"ok\":1}" : "{\"ok\":0}";
  }
  if(path == "/trim"){
    if(a.has("j") && a.has("d")){
      int j = a.get("j").toInt();
      if(j < 0 || j >= NUM_SERVOS) return "{\"ok\":0,\"error\":\"no such joint\"}";
      TRIM[j] = constrain(a.get("d").toFloat(), -TRIM_MAX, TRIM_MAX);
      saveTrim(j);
      if(attached[j]){ lastServoUs[j] = -1; lastServoWriteMs[j] = 0; writeServo(j); }   // takes effect now
    }
    String s = "{\"ok\":1,\"trim\":[";
    for(int j=0;j<NUM_SERVOS;j++){ if(j) s += ","; s += String(TRIM[j], 2); }
    return s + "]}";
  }
  if(path == "/power"){
    bool on = a.get("on").toInt()==1;
    if(a.has("j")){
      int j=a.get("j").toInt();
      if(j==NUM_SERVOS) powerStepper(on); else powerServo(j,on);
    } else {
      for(int i=0;i<NUM_SERVOS;i++) powerServo(i,on);
      powerStepper(on);
    }
    return "{\"ok\":1}";
  }
  return "{\"ok\":0,\"error\":\"unknown command\"}";
}

const char* CMD_PATHS[] = {"/state", "/set", "/setall", "/stop", "/speed", "/home", "/basezero", "/power", "/tel", "/ver", "/trim", "/diag", "/rest", "/relax"};

void setupServer(){
  server.on("/",[](){ server.send_P(200,"text/html",PAGE); });
  for(const char* p : CMD_PATHS){
    server.on(p, [p](){ Args a = argsFromServer(); server.send(200, "application/json", handleCmd(String(p), a)); });
  }
  server.begin();
}

// ---------------- cable input, read without blocking ----------------
uint8_t xorOf(const String& s, int n){ uint8_t x = 0; for(int i=0;i<n;i++) x ^= (uint8_t)s[i]; return x; }

String hex2(uint8_t v){ const char* h = "0123456789ABCDEF"; String s; s += h[v >> 4]; s += h[v & 15]; return s; }

void handleLine(Stream& port, String line){
  line.trim();
  if(line.length() == 0) return;
  String tail = "";                                  // "~n" echoed back when the line was checked
  int star = line.lastIndexOf('*');
  if(star >= 0 && star == (int)line.length() - 3){
    long want = strtol(line.substring(star + 1).c_str(), NULL, 16);
    if(xorOf(line, star) != want){
      String r = "{\"ok\":0,\"error\":\"checksum\"}";
      port.println(r + "*" + hex2(xorOf(r, r.length())));
      return;
    }
    line = line.substring(0, star);
    int tl = line.lastIndexOf('~');
    if(tl >= 0){ tail = line.substring(tl); line = line.substring(0, tl); }
    tail += "*";                                     // marks "answer with a check"
  }
  if(line[0] == '/'){
    int q = line.indexOf('?');
    String path = q < 0 ? line : line.substring(0, q);
    if(tail.length() == 0 && &port != (Stream*)&Serial && path != "/ver" && path != "/tel"){
      String r = "{\"ok\":0,\"error\":\"unchecked\"}";        // only checked lines may move the arm
      port.println(r + "*" + hex2(xorOf(r, r.length())));
      return;
    }
    Args a = q < 0 ? Args() : argsFromQuery(line.substring(q + 1));
    if(path == "/telstream"){
      telHz = constrain(a.get("hz").toFloat(), 0.0f, 100.0f);
      telPort = telHz > 0 ? &port : NULL;
      sendChecked(port, "{\"ok\":1}", tail);
      return;
    }
    sendChecked(port, handleCmd(path, a), tail);
    return;
  }
  // The old typed commands (help, get, "1 90", on/off, home...) are accepted ONLY from the USB serial monitor.
  // On the Pi cable anything that is not a /command is ignored, so stray text (a boot message, line noise)
  // can never be read as "move joint 1".
  if(&port == (Stream*)&Serial) legacyLine(line);
}

void sendChecked(Stream& port, const String& reply, const String& tail){
  if(tail.length() == 0){ port.println(reply); return; }          // an unchecked command gets a plain answer
  String body = reply + tail.substring(0, tail.length() - 1);      // json~n
  port.println(body + "*" + hex2(xorOf(body, body.length())));
}

void pollPort(Stream& port, String& buf){
  while(port.available()){
    char c = (char)port.read();
    if(c == '\n' || c == '\r'){
      if(buf.length()){ String line = buf; buf = ""; handleLine(port, line); }
    } else if(buf.length() < 400) buf += c;
  }
}

void streamTelemetry(){
  if(telHz <= 0 || telPort == NULL) return;
  uint32_t now = millis();
  if(now - lastTelMs < (uint32_t)(1000.0f / telHz)) return;
  lastTelMs = now;
  telPort->println(telJson());
}

// ---------------- setup/loop ----------------
void setup(){
  Serial.begin(115200); delay(300);
  loadTrims();                                   // before any servo is attached
  PiSerial.setRxBufferSize(1024);
  PiSerial.begin(PI_UART_BAUD, SERIAL_8N1, PI_UART_RX, PI_UART_TX);
  for(int i=0;i<NUM_SERVOS;i++) seg[i].active = false;
  ESP32PWM::allocateTimer(0); ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2); ESP32PWM::allocateTimer(3);
  for(int i=0;i<NUM_SERVOS;i++){
    currentAngle[i]=REST_POS[i]; targetAngle[i]=REST_POS[i]; attached[i]=false;   // the arm sits at rest
  }
  for(int i=0;i<NUM_SERVOS;i++){                 // shoulder first (lifts the arm clear), gripper last
    if(START_ATTACHED || wakeOnBoot){ powerServo(i,true); delay(250); }
  }
  pinMode(EN_PIN,OUTPUT);
  if(stepCore == 0){                             // interrupts are set up on the core that connects the stepper
    xTaskCreatePinnedToCore(stepperSetupTask, "stsetup", 4096, NULL, 5, NULL, 0);
    for(int t=0; t<2000 && !stepperReady; t++) delay(1);
  } else connectStepper();
  Serial.printf("Base step driver: %s on core %d, servo writes every %d ms\n", baseDrv, stepCore, servoWriteMs);
  restoreBasePos();                              // the base is where it was when the power went off
  Serial.printf(baseRestored ? "Base position restored: %.1f deg\n" : "No saved base position (%.1f): run basehome\n", baseAngle());
  powerStepper(STEPPER_START_ENABLED);
  Serial.println((START_ATTACHED || wakeOnBoot)?"\nServos ON, holding the rest pose.":"\nSAFE START: everything OFF (commanded to the rest pose, so switching on will not fling).");

  bool onHome=false;
  if(USE_HOME_WIFI){
    WiFi.mode(WIFI_STA); WiFi.setSleep(false);   // no WiFi power saving (it caused 200 ms answer delays)
    WiFi.begin(HOME_SSID,HOME_PASSWORD);
    Serial.printf("Joining \"%s\"",HOME_SSID);
    for(int t=0;t<40 && WiFi.status()!=WL_CONNECTED;t++){ delay(400); Serial.print("."); }
    if(WiFi.status()==WL_CONNECTED){ onHome=true;
      Serial.printf("\nConnected! http://%s\n",WiFi.localIP().toString().c_str()); }
    else Serial.println("\nHome WiFi failed; falling back to AP.");
  }
  if(!onHome){
    WiFi.mode(WIFI_AP); WiFi.softAP(AP_SSID,AP_PASSWORD);
    Serial.printf("AP \"%s\" (pass %s) -> http://192.168.4.1\n",AP_SSID,AP_PASSWORD);
  }
  setupServer();
  // Wireless uploads (Arduino IDE: Tools > Port > arm-esp32 at <ip>). When an upload starts, every joint is
  // told to hold still; the ESP32 then restarts, and the servos go limp until they are switched on again.
  ArduinoOTA.setHostname(OTA_NAME);
  ArduinoOTA.setPassword(OTA_PASSWORD);
  ArduinoOTA.onStart([](){
    for(int i=0;i<NUM_SERVOS;i++){ targetAngle[i]=currentAngle[i]; glideStep[i]=0; seg[i].active=false; }
    jogDps = 0; jogDir = 0;
    if(stepper) stepper->stopMove();
  });
  ArduinoOTA.begin();
  Serial.println("Serial: type 'help'.  Cable commands are URL paths, e.g. /tel");
}

void loop(){
  if(relaxOffAt && (int32_t)(millis() - relaxOffAt) >= 0){    // /relax arrived: servos off
    bool still = true;
    for(int i=0;i<NUM_SERVOS;i++) if(seg[i].active) still = false;
    if(still){ for(int i=0;i<NUM_SERVOS;i++) powerServo(i, false); relaxOffAt = 0; }
  }
  if(restartAt && (int32_t)(millis() - restartAt) >= 0){   // a core=/drv= switch was asked for
    if(stepper && !stepper->isRunning()) saveBasePos(stepper->getCurrentPosition());
    ESP.restart();
  }
  readShoulderTemp();
  uint32_t nowUs = micros();                       // watch for long gaps between loop runs
  if(loopLastUs && nowUs - loopLastUs > loopMaxUs) loopMaxUs = nowUs - loopLastUs;
  loopLastUs = nowUs;
  server.handleClient(); ArduinoOTA.handle(); pollPort(Serial, lineUsb); pollPort(PiSerial, linePi);
  updateServos(); updateBaseJog(); sleepDriverWhenIdle(); rememberBase(); streamTelemetry();
  uint32_t pauseStart = millis();                  // the same 10 ms pause, but the Pi cable is answered during it
  do { pollPort(PiSerial, linePi); delay(1); } while(millis() - pauseStart < 10);
}

void updateServos(){
  uint32_t now = micros();
  for(int i=0;i<NUM_SERVOS;i++){
    if(!attached[i]) continue;
    if(seg[i].active){                       // time-based move
      bool done = false;
      float p = segPos(seg[i], now, done);
      if(done){ p = seg[i].p1; seg[i].active = false; }
      currentAngle[i] = p;
      continue;
    }
    float d=targetAngle[i]-currentAngle[i];
    if(fabs(d)<0.01) continue;
    float st=(glideStep[i]>0)?glideStep[i]:degPerStep[i];
    if(fabs(d)<=st) currentAngle[i]=targetAngle[i];
    else currentAngle[i]+=(d>0?st:-st);
  }
  for(int i=0;i<NUM_SERVOS;i++) writeServo(i);      // one place writes the servos (see writeServo)
}

void legacyLine(String line){
  line.trim(); line.toLowerCase();
  if(line.length()==0) return;
  if(line=="help"){
    Serial.println(" 1-5 <ang> | 6 <ang -180..180> | on/off <1-6> | home | basezero");
    Serial.println(" get | speed <dps>"); return; }
  if(line=="get"){
    for(int i=0;i<NUM_SERVOS;i++)
      Serial.printf(" %d %-13s %3d->%3d %s\n",i+1,JOINT_NAMES[i],
        (int)currentAngle[i],(int)targetAngle[i],attached[i]?"":"(OFF)");
    Serial.printf(" 6 Base          %4d      %s %s\n",(int)baseAngle(),
      stepperEnabled?"":"(OFF)",baseHomed?"":"(NOT HOMED)");
    return; }
  if(line=="home"){ for(int i=0;i<NUM_SERVOS;i++) targetAngle[i]=HOME_POS[i]; return; }
  if(line=="basezero"){ baseSetZero(); Serial.println("Base zeroed here."); return; }
  int sp=line.indexOf(' ');
  String cmd=(sp==-1)?line:line.substring(0,sp);
  String arg=(sp==-1)?"":line.substring(sp+1); arg.trim();
  if(cmd=="speed"){
    float v=constrain(arg.toFloat(),1.0f,2000.0f);
    for(int i=0;i<NUM_SERVOS;i++) degPerStep[i]=speedBase[i]*(v/100.0);
    return; }
  if(cmd=="on"||cmd=="off"){
    int j=arg.toInt()-1; bool on=(cmd=="on");
    if(j==NUM_SERVOS) powerStepper(on);
    else powerServo(j,on);
    return; }
  int j=cmd.toInt()-1;
  if(j>=0 && j<NUM_SERVOS && arg.length()>0) setServoTarget(j,arg.toInt());
  else if(cmd.toInt()==6 && arg.length()>0) baseGoTo(arg.toFloat());
}
