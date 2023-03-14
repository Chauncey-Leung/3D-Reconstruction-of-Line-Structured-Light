#include "pwm.h"
#include "stm32f4xx.h"

//arr：自动重装值 psc：时钟预分频数
void  TIM14_PWM_Init (uint16_t arr,uint16_t psc)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
	TIM_OCInitTypeDef TIM_OCInitStructure;

	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM14,ENABLE);		//TIM14 时钟使能
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOF, ENABLE);  	//GPIOF	时钟使能
	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9; //GPIOF9
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF; //复用功能
//	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; //速度 50MHz
//	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP; //推挽复用输出
//	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_UP; //上拉
	GPIO_Init(GPIOF, &GPIO_InitStructure); //初始化 PF9
	GPIO_PinAFConfig(GPIOF,GPIO_PinSource9, GPIO_AF_TIM14); //GPIOF9 复用为定时器 14
	
	TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1;
	TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_TimeBaseStructure.TIM_Period = arr;
	TIM_TimeBaseStructure.TIM_Prescaler = psc;
	TIM_TimeBaseInit(TIM14, &TIM_TimeBaseStructure); //根据指定的参数初始化 TIM14 的
	
	//初始化 TIM14 Channel1 PWM 模式
	TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1; //PWM 调制模式 1
	TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable; //比较输出使能
	TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_Low; //输出极性低
	TIM_OC1Init(TIM14, &TIM_OCInitStructure); //初始化外设 TIM1 4OC1
	TIM_OC1PreloadConfig(TIM14, TIM_OCPreload_Enable); //使能预装载寄存器
	TIM_ARRPreloadConfig(TIM14,ENABLE);//ARPE 使能
	TIM_Cmd(TIM14, ENABLE); //使能 TIM14 
}
