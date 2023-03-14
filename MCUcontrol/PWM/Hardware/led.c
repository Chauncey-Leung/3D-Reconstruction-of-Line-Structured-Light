#include "stm32f4xx.h"
#include "led.h"

/*
*--------------------------------------------------------------------------------------------------------
* Function:          led_init
* Description:       ST-M3-LITE-407VE精简开发板有1个引脚连接了LED发光二极管，
                     GPIOC.2--LED1，该函数的作用是对其各引脚进行初始化。
* Input:             none
* Output:            none
* Return:            none
* Created by:        alvan
* Created date:      2014-7-29
* Others:        	
*---------------------------------------------------------------------------------------------------------
*/

void  led_init (void)
{ 
	
    GPIO_InitTypeDef GPIO_InitStructure;
    
    RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);    /* 使能端口PORTC时钟                                  */  	 
    
    GPIO_InitStructure.GPIO_Pin   = GPIO_Pin_5|GPIO_Pin_6;               
    GPIO_InitStructure.GPIO_Speed  = GPIO_Speed_50MHz;
		GPIO_InitStructure.GPIO_Mode   = GPIO_Mode_OUT;
    GPIO_InitStructure.GPIO_OType  = GPIO_OType_PP;
		GPIO_InitStructure.GPIO_PuPd   = GPIO_PuPd_UP;
    GPIO_Init(GPIOE, &GPIO_InitStructure);                /* 配置引脚GPIOC.2为推挽输出,IO最大速度50M            */
	
    GPIO_ResetBits(GPIOE , GPIO_Pin_5|GPIO_Pin_6);   
}











