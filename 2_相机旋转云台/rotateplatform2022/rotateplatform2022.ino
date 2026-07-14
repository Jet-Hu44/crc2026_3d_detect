#include <TM1638plus.h>
#include <Servo.h>
#include <EEPROM.h>
// GPIO I/O pins on the Arduino connected to strobe, clock, data,
//pick on any I/O you want.
#define  STROBE_TM 4 // strobe = GPIO connected to strobe line of module
#define  CLOCK_TM 6  // clock = GPIO connected to clock line of module
#define  DIO_TM 7 // data = GPIO connected to data line of module
bool high_freq = false; //default false,, If using a high freq CPU > ~100 MHZ set to true. 
Servo myservo; 
//Constructor object (GPIO STB , GPIO CLOCK , GPIO DIO, use high freq MCU)
TM1638plus tm(STROBE_TM, CLOCK_TM , DIO_TM, high_freq);

String inputString = "";         // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete

// Some vars and defines for the tests.
#define myTestDelay  5000
#define myTestDelay1 1000
uint8_t  testcount = 14;


uint8_t buttons,last_buttons;
int button_com=0;
#define RUNING 1
#define RECORD 2

int runF = RUNING;
byte angles[3]={0,0,0};
int recordCount = 0;
int curstep = 90;
void setup()
{
  inputString.reserve(200);
  //Serialinit();
  Serial.begin(9600);
  tm.displayBegin();
  delay(myTestDelay1);
  int word_address = 0;

  for(int i=0;i<3;i++)
  {
    angles[i] = EEPROM.read(word_address);
    word_address++;
  }
  //curstep = angles[2];
  myservo.attach(9); 

          tm.displayText("P");
        tm.displayHex(1, 0);
        tm.displayHex(2, 0);
        tm.displayHex(3, 0);
//EEPROM.write(word_address,0x7F);
  //Test 0 reset
  //Test0();
}
void rotate(int step)
{
  if(step < 0 )
      step = 0;
  else if(step > 180)
      step = 180;
  if(curstep<=step)
    for(;curstep<=step;curstep++)
    {
       myservo.write(curstep);
       delay(35);
    }
  else if(curstep > step)
    for(;curstep>step;curstep--)
    {
       myservo.write(curstep);
       delay(35);

    }
    curstep = step;
}
void rotateL()
{
  curstep--;

        rotate(curstep);
}
void rotateR()
{
          curstep++;
        if(curstep > 180 )
          curstep = 180;
        rotate(curstep);
}
void loop()
{
  int word_address = 0;
  uint8_t buttons = tm.readButtons();
  if (stringComplete) {
    if(inputString == "1\n")
        button_com = 1;
    if(inputString == "2\n")
        button_com = 2;
    if(inputString == "3\n")
        button_com = 3;
    Serial.println(button_com);
    inputString = "";
    stringComplete = false;
  }
  else
  {
    button_com = -1;
  }
  if(last_buttons==0)
  {
    //处理第4按钮，0x20，modechange
    if(buttons==0x20)
    {
      if(runF==RUNING)
      {
        runF = RECORD;
        recordCount = 0;
        tm.displayText("R");
        tm.displayHex(1, 0);
        tm.displayHex(2, 0);
        tm.displayHex(3, 0);
      }
      else
      {
        tm.displayText("P");
        runF = RUNING;
                tm.displayHex(1, 0);
        tm.displayHex(2, 0);
        tm.displayHex(3, 0);
      }
    }
    if(runF == RUNING)
    {
      
      if(buttons==0x01||button_com == 1)
      {
        rotate(angles[0]);
        tm.displayHex(1, 1);
        tm.displayHex(2, 0);
        tm.displayHex(3, 0);
      }
      if(buttons==0x10||button_com == 2)
      {
        rotate(angles[1]);
        tm.displayHex(1, 0);
        tm.displayHex(2, 1);
        tm.displayHex(3, 0);
      }
      if(buttons==0x02||button_com == 3)
      {
        rotate(angles[2]);
        tm.displayHex(1, 0);
        tm.displayHex(2, 0);
        tm.displayHex(3, 1);
      }
    }
    else if(runF == RECORD)
    {
      if(buttons==0x01)
      {

        if(recordCount < 3)
        {
          //rotate(angles[0]);
          EEPROM.write(recordCount,curstep);
          angles[recordCount] = curstep;
          recordCount++;
          for(int i = 0;i<recordCount;i++)
          {
            tm.displayASCII(i+1, '1');
          }
        }
      }   
    } 
    
  } 
  if(runF==RECORD&&(buttons==0x10||buttons==0x02))
  {
       if(buttons==0x10)
      {
        rotateL();
      }
      if(buttons==0x02)
      {
        rotateR();
      }      
  }
  last_buttons = buttons;
  delay(100);
//byte value;
//value = EEPROM.read(word_address);
//Serial.println(value,HEX);
}
void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // add it to the inputString:
    inputString += inChar;
    // if the incoming character is a newline, set a flag so the main loop can
    // do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}
