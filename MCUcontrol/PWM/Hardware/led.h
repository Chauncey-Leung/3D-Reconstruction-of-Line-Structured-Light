#ifndef  __LED_H
#define  __LED_H
	
#define	digitalHi(p,i)			{p->BSRR=i;}			//设置为高电平		
#define digitalLo(p,i)			{p->BRR=i;}				//输出低电平
#define digitalToggle(p,i)		{p->ODR ^=i;}			//输出反转状态


#define led1_off()        GPIO_SetBits(GPIOE, GPIO_Pin_5)		
#define led1_on()       GPIO_ResetBits(GPIOE, GPIO_Pin_5)

#define led2_off()        GPIO_SetBits(GPIOE, GPIO_Pin_6)		
#define led2_on()       GPIO_ResetBits(GPIOE, GPIO_Pin_6)
//#define LED_TOGGLE			 

	
void  led_init (void);


#endif /*__LED_H*/


