#include "stm32f4xx.h"
#include "usart.h"
#include "delay.h"
#include "led.h"
#include "pwm.h"

#define DUTYCYCLE 0.5
#define ARR	1000
#define PSC	3360
//��ʱ��ʱ��Ϊ 84M����Ƶϵ��Ϊ 3360,��װ��ֵ 1000
//���� PWM Ƶ��Ϊ 84M / (1000 * 3360) = 25Hz

int main(void)
{
	uart_init(115200);
	delay_init(84);

	TIM14_PWM_Init(ARR-1,PSC-1); 
	
	//�޸ĵ���CCR1
	TIM_SetCompare1(TIM14,ARR * DUTYCYCLE); //�޸ıȽ�ֵ���޸�ռ�ձ�
  while(1);
}
