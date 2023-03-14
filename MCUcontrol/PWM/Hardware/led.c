#include "stm32f4xx.h"
#include "led.h"

/*
*--------------------------------------------------------------------------------------------------------
* Function:          led_init
* Description:       ST-M3-LITE-407VE���򿪷�����1������������LED��������ܣ�
                     GPIOC.2--LED1���ú����������Ƕ�������Ž��г�ʼ����
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
    
    RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);    /* ʹ�ܶ˿�PORTCʱ��                                  */  	 
    
    GPIO_InitStructure.GPIO_Pin   = GPIO_Pin_5|GPIO_Pin_6;               
    GPIO_InitStructure.GPIO_Speed  = GPIO_Speed_50MHz;
		GPIO_InitStructure.GPIO_Mode   = GPIO_Mode_OUT;
    GPIO_InitStructure.GPIO_OType  = GPIO_OType_PP;
		GPIO_InitStructure.GPIO_PuPd   = GPIO_PuPd_UP;
    GPIO_Init(GPIOE, &GPIO_InitStructure);                /* ��������GPIOC.2Ϊ�������,IO����ٶ�50M            */
	
    GPIO_ResetBits(GPIOE , GPIO_Pin_5|GPIO_Pin_6);   
}











