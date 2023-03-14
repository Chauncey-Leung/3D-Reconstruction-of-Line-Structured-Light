#include "stm32f4xx.h"
#include "usart.h"
#include "delay.h"
#include "led.h"
#include "pwm.h"

#define DUTYCYCLE 0.5
#define ARR	1000
#define PSC	3360
//定时器时钟为 84M，分频系数为 3360,重装载值 1000
//所以 PWM 频率为 84M / (1000 * 3360) = 25Hz

int main(void)
{
	uart_init(115200);
	delay_init(84);

	TIM14_PWM_Init(ARR-1,PSC-1); 
	
	//修改的是CCR1
	TIM_SetCompare1(TIM14,ARR * DUTYCYCLE); //修改比较值，修改占空比
  while(1);
}
