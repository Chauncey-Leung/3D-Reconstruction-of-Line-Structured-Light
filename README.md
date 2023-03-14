# 3D-Reconstruction-of-Line-Structured-Light

	coordinate.txt
  
				输出的三维坐标
	
  
	/Image
  
  
		为工程中存储图片的文件夹。

			/Image/CameraCalibration
      
				为相机内外参数标定所使用图片。
			
      
			/Image/LightPlaneCalibration
      
				为光平面标定使用文件夹。
        
				将 Cali_temp_las.jpg、Cali_world_las.jpg 两张图片提取中心线(XXX中心线提取程序)并将结果结果存入相应文件夹中。
				注意在相机标定时需要有与这两张照片相应的图片，并明确其标定的序号，体现在程序中的World和Temp。
				结果：
        
					中心线坐标存储在 CenterPoint_f 文件夹中
          
					中心线图像存储在 img_line 文件夹中
          
					带有原图的中心线图像存储在 img_res 文件夹中
				
        
			/Image/RotationAxisCalibration
      
				存储旋转轴标定所需图片
				
        
			/Image/Scan
      
				存储扫描图像结果
        
				/Image/Scan/RawImage 中存储扫描得到的原始图片，其中MyRame.bat用于统一命名，可用记事本修改。
        
				/Image/Scan/BellImage 中存储经畸变矫正(ImageCorrection.m)后的图片。
        
				/Image/Scan/CenterPoint_f 将已矫正的图片经中心线提取(XXX中心线提取程序)的中心线坐标存储在其中。
				
				
			MyCheckerBoard.docx 棋盘格图像，可以修改大小打印
		
		
	/MCUcontrol
  
			为单片机驱动步进电机的程序，实为PWM脉冲输出程序。
      
			/MCUcontrol/PWM 为STM32F407程序，可供参考，也可以用任意别的单片机实现相应功能。
      
			其他文件为必要的参考手册。
		
		
	/MatlabCode
  
			Matlab代码，实现了光平面标定、三维重建等功能。
      
			详情见代码内说明。
		
		
	/PythonCode
  
			Python代码，实现了中心线提取、旋转轴标定等功能。
      
			详情见代码内说明。
		
		
		
程序说明

	由于使用Matlab和Python进行整个工程的实现，为清晰工程使用思路，特此说明。
  
	1.相机的标定
  
		打印所需棋盘格。
		固定好整个系统。
		调整相机焦距，是在所需位置成像清晰。
		将棋盘格摆放各种角度，并拍照，将结果存入相应文件夹。
		使用Matlab 自带App “Camera Calibrator”进行标定，导出结果并保存。
		同时要拍两张带有激光的图片，在上述图片中应有与之相应的图片。
		
	2.光平面标定准备工作
  
		将两张带有激光的图片进行中心线提取(XXX中心线提取程序)，将结果存入相应文件夹。
		
	3.图像扫描获取
  
		将单片机管脚与电机连接好，连接关系如下：
    
			单片机			电机
      
			F9		————	PUL
      
			+5V		————	DIR
      
		使用电脑软件 “相机” 录取扫描图像，并使用Premiere Pro 将视频转换成图片格式，存入相应的文件夹。
	
	4.图像的去畸变
  
		运行 ImageCorrection.m 将扫描得到图片进行去畸变和简单的切割，存入相应的文件夹。
	
	5.扫描图像的中心线提取
  
		对上述图像提取中心线，并存入相应文件夹。
	
	6.旋转轴标定
  
		为了标定旋转轴，需要获取一点旋转一圈的坐标，由于视角限制，仅扫描获取约60°的图片来拟合。
    
		将棋盘格固定在载物台上旋转，获得其一圈的坐标。
    
		标定出旋转轴。
		
	7.光平面标定与三维重建
  
		(main.m)
    
		将相机标定结果导入。
    
		将带有激光图片的中心线提取坐标导入。
    
		光平面标定得到光平面参数 a,b,c,d
    
		导入提取好的扫描图像中心线坐标，利用光平面方程和内外参数将{pixel}坐标转换为{world}坐标。
    
		（可先将坐标绕Y轴（也可能是x轴，需要看标定时世界坐标系的旋转轴是哪个轴）旋转，导出建模观察效果）
    
		将标定的旋转轴导入。
    
		将坐标绕旋转轴旋转，导出三维坐标数据。
		
		将导出的数据导入catia（本项目使用catia，使用任意三维建模软件均可）
    
		
