#include "stm32f4xx.h"
#include "key.h"

void Delay(__IO u32 nCount)	 //简单的延时函数
{
	for(; nCount != 0; nCount--);
} 
/*
*--------------------------------------------------------------------------------------------------------
* Function:       key_init
* Description:    
* Input:          none
* Output:         none
* Return:         none
* Created by:     alvan 
* Created date:   2014-07-29
* Others:        	
*---------------------------------------------------------------------------------------------------------
*/
void  key_init (void)
{ 
	GPIO_InitTypeDef GPIO_InitStructure;
	
    RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);  /* 使能端口时钟                        */ 

    GPIO_InitStructure.GPIO_Pin   = GPIO_Pin_4;	                             
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
    GPIO_InitStructure.GPIO_Mode  = GPIO_Mode_IN;
    GPIO_Init(GPIOE, &GPIO_InitStructure);                 /* 配置引脚为上拉输入 	*/	 
	
	
		RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);  /* 使能端口时钟                        */ 

    GPIO_InitStructure.GPIO_Pin   = GPIO_Pin_0;	                             
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
    GPIO_InitStructure.GPIO_Mode  = GPIO_Mode_IN;
    GPIO_Init(GPIOA, &GPIO_InitStructure);    
}


/*
 * 函数名：Key_Scan
 * 描述  ：检测是否有按键按下
 * 输入  ：无	
 * 输出  ：返回按键值
 */

unsigned char KEY_Scan(void)
{	 
	unsigned char key_code;
  if(KEY1_Scan()==1)key_code=1;
	else if(KEY2_Scan()==1)key_code=2;
	else key_code=0;
  return key_code;
}

unsigned char KEY1_Scan(void)
{	 
	static char key_up1=0;//按键按松开标志	
 	if(KEY1==0)
	{
		/*延时去抖动 */
	  Delay(10000);	
		if(KEY1==0)
		{
			key_up1=1;
    }		
	 }
	 if(KEY1==1&&key_up1==1)
	 {
		key_up1=0;
    return 1;
    }
		return 0;
}

unsigned char KEY2_Scan(void)
{	 
	static char key_up1=0;//按键按松开标志	
 	if(KEY2==1)
	{
		/*延时去抖动 */
	  Delay(10000);	
		if(KEY2==1)
		{
			key_up1=1;
    }		
	 }
	 if(KEY2==0&&key_up1==1)
	 {
		key_up1=0;
    return 1;
    }
		return 0;
}







