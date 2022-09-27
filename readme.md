## Deep learning ECG classification

### dataset
 
- The data set from [physionet](https://physionet.org/content/ptb-xl/1.0.2/example_physionet.py)
- Download the files using your terminal: 
  ```shell
  wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.2/
  ```
- this project process with file :
  ```  
  ./Data/Dataloader.py
  ```
- we provide four kinds of dataset 
  ```
  shape(None,12,1000)
  
  shape(None,1000,12)
  
  shape(None,12,1000,1)
  
  shape(None,1000,12,1)
  ```