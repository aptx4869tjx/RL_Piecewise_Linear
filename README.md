## This is the code repository for **Boosting Verification of Deep Reinforcement Learning via Piece-wise Linear Decision Neural Networks**.

------------
We provide the Flow* models evaluated in our experiments. See the following directory:
```
cd models/
```

Then one can analyze these models through Flow*. The Installation of Flow* can be found in <https://github.com/chenxin415/flowstar>.
The commands for analyzing these models is:
```
./flowstar < modelfile
```

------------
## Installation
If one want to train PLDNNs and output the corresponding models, it requires the installation of our python project.
```
conda create -n pldnn python=3.7

conda activate pldnn

pip install -r requirements.txt
```
------------
## Training PLDNNs and output Flow* model
run either:
```
cd abstract/b1/
python b1_abs.py

cd abstract/b2/
python b2_abs.py

cd abstract/b3/
python b3_abs.py

cd abstract/b4/
python b4_abs.py

cd abstract/b5/
python b5_abs.py

cd abstract/tora/
python tora_abs.py

cd abstract/cartpole/
python cartpole_abs.py

cd abstract/quad/
python quad_abs.py
```


