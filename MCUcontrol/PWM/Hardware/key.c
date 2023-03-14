#include "stm32f4xx.h"
#include "key.h"

void Delay(__IO u32 nCount)	 //�򵥵���ʱ����
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
	
    RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOE, ENABLE);  /* ʹ�ܶ˿�ʱ��                        */ 

    GPIO_InitStructure.GPIO_Pin   = GPIO_Pin_4;	                             
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
    GPIO_InitStructure.GPIO_Mode  = GPIO_Mode_IN;
    GPIO_Init(GPIOE, &GPIO_InitStructure);                 /* ��������Ϊ�������� 	*/	 
	
	
		RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);  /* ʹ�ܶ˿�ʱ��                        */ 

    GPIO_InitStructure.GPIO_Pin   = GPIO_Pin_0;	                             
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
    GPIO_InitStructure.GPIO_Mode  = GPIO_Mode_IN;
    GPIO_Init(GPIOA, &GPIO_InitStructure);    
}


/*
 * ��������Key_Scan
 * ����  ������Ƿ��а�������
 * ����  ����	
 * ���  �����ذ���ֵ
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
	static char key_up1=0;//�������ɿ���־	
 	if(KEY1==0)
	{
		/*��ʱȥ���� */
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
	static char key_up1=0;//�������ɿ���־	
 	if(KEY2==1)
	{
		/*��ʱȥ���� */
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







