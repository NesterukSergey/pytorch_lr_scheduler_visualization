# Pytorch learning rate scheduler visualization

![UI sample](https://github.com/NesterukSergey/pytorch_lr_scheduler_visualization/blob/master/images/UI_sample.jpg?raw=true)

This repo contains simple code for visualizing popular learning rate schedulers.

The interactive interface allows to alter schedulers parameters and plot them on one canvas. Additionally, underlying Pytorch code to reproduce your tuned scheduler is generated. This is aimed to help forming an intuition for setting lr scheduler in your DL project.



## Run interactively

To run the code with interactive Web interface:

`git clone https://github.com/NesterukSergey/pytorch_lr_scheduler_visualization.git`  
`cd pytorch_lr_scheduler_visualization`  
`python3 -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`  
`cd streamlit_server/`  
`streamlit run __main__.py`  

This will run streamlit server (default adress is: http://localhost:8501/). You can access it in your browser. 


## Run in jupyter notebook

To run only default lr schedulers behaviour:
`git clone https://github.com/NesterukSergey/pytorch_lr_scheduler_visualization.git`  
`cd pytorch_lr_scheduler_visualization`  
`python3 -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements-noweb.txt`  

And launch Demo.ipynb

