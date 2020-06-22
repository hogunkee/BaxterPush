# robosuite-gqcnn
robosuite env that works with gqcnn.
gqcnn/model, gqcnn/data, and robosuite/robosuite/gqcnn/GQ-Image-Wise/train_indices_image_wise.pkl are deliberately deleted due to the max file size of Github. Please refer to the original project page for these files.

# Installation
First, make a virtual environment with the saved configuration.

	conda env create -f environment.yml
	conda activate robosuite

	cd gqcnn
	pip install -e .
	cd ..

	cd robosuite
	pip install -e .
	cd robosuite/perception
	pip install -e .

# 6-DoF Grapsing data collecting
Randomly select the robot arm approaching angle within Baxter configuration space and try grasp with the best grasp from pre-trained GQCNN at the approaching angle. 

How to run:

	cd /robosuite-gqcnn/robosuite/robosuite/scripts	
	python demo_baxter_6DoF_data_collecting.py --seed 0 --num-objects 5 --num-episodes 100 --num-steps 5 --render True --bin-type "table" --object-type "T"

	seed (default 0): random seed number

	num_objects (default 5): the number of objects spawned in environments

	num_episodes (default 100): the number of episodes of collecting grasping data
	In every episode, objects and environments will be initialized

	num_steps (default 5): the number of grasp trials in each episode

	render (default True): if render is true, MuJoCo evironment will be rendered.

	bin_type (default "table"): "table" or "bin"

	object_type (default "T"): "T", "L" or "3DNet"


# viewpoint data collecting
Using CEM method, get the highest quality viewpoint angle

How to run:

	cd /robosuite-gqcnn/robosuite/robosuite/scripts	
	python demo_baxter_viewpoint_data_collecting.py --seed 0 --num-objects 5 --num-episodes 100 --render True --bin-type "table" --object-type "T" --test True --config-file

	seed (default 0): random seed number

	num_objects (default 5): the number of objects spawned in environments

	num_episodes (default 100): the number of episodes of collecting grasping data
	In every episode, objects and environments will be initialized


	render (default True): if render is true, MuJoCo evironment will be rendered.

	bin_type (default "table"): "table" or "bin"

	object_type (default "T"): "T", "L" or "3DNet"

	test (default False): When False, collect viewpoint data. When True, test the trained network

	config_file (default "config_example.yaml"): config file of the trained network. Please correct model_dir and data_dir
