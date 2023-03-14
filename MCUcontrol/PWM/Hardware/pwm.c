#include "pwm.h"
#include "stm32f4xx.h"

//arr���Զ���װֵ psc��ʱ��Ԥ��Ƶ��
void  TIM14_PWM_Init (uint16_t arr,uint16_t psc)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
	TIM_OCInitTypeDef TIM_OCInitStructure;

	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM14,ENABLE);		//TIM14 ʱ��ʹ��
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOF, ENABLE);  	//GPIOF	ʱ��ʹ��
	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9; //GPIOF9
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF; //���ù���
//	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; //�ٶ� 50MHz
//	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP; //���츴�����
//	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_UP; //����
	GPIO_Init(GPIOF, &GPIO_InitStructure); //��ʼ�� PF9
	GPIO_PinAFConfig(GPIOF,GPIO_PinSource9, GPIO_AF_TIM14); //GPIOF9 ����Ϊ��ʱ�� 14
	
	TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1;
	TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_TimeBaseStructure.TIM_Period = arr;
	TIM_TimeBaseStructure.TIM_Prescaler = psc;
	TIM_TimeBaseInit(TIM14, &TIM_TimeBaseStructure); //����ָ���Ĳ�����ʼ�� TIM14 ��
	
	//��ʼ�� TIM14 Channel1 PWM ģʽ
	TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1; //PWM ����ģʽ 1
	TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable; //�Ƚ����ʹ��
	TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_Low; //������Ե�
	TIM_OC1Init(TIM14, &TIM_OCInitStructure); //��ʼ������ TIM1 4OC1
	TIM_OC1PreloadConfig(TIM14, TIM_OCPreload_Enable); //ʹ��Ԥװ�ؼĴ���
	TIM_ARRPreloadConfig(TIM14,ENABLE);//ARPE ʹ��
	TIM_Cmd(TIM14, ENABLE); //ʹ�� TIM14 
}