


#ifndef  __KEY_H
#define  __KEY_H

#define KEY1 GPIO_ReadInputDataBit(GPIOE,GPIO_Pin_4)	//读PA15
#define KEY2 GPIO_ReadInputDataBit(GPIOA,GPIO_Pin_0)	//读PA15 
unsigned char KEY2_Scan(void);
void Key_GPIO_Config(void);//IO初始化

unsigned char KEY1_Scan(void);  //按键扫描函数


#endif


