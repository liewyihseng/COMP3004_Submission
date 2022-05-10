
# Image-based Reinforcement Learning with CarRacing-V0


## User Manual
#### Cloning of this Repository on Github
* Using the Command Line Tool in your desired IDE, run:

		git clone https://github.com/liewyihseng/COMP3004_Submission.git
* This will allow the latest version of source code to be cloned into the workspace.
* If you are facing any issue on cloning this file, do drop me an email at liewyihseng177@gmail.com as this repository is currently still in private mode.

### Importing of Anaconda Environment
* Have Anaconda Navigator opened in your machine.
* Head to the Environments tab on the most left part of the window.
* Search for 'Import' that lies at the bottom left part of the window.
* At the 'Local Drive', simply insert the path that directs to environment.yml file within the cloned repository.
* Simply assign a name for the new environment and remember to check the 'Overwrite existing environment' checkbox.
* After that, simply select 'Import'.
* The process of importing the environment might take awhile, please be patient.
* If you have followed the steps, the designated environment with the environment name you have specified has been imported.

# File Structure
## External
### csv_mean_reward
* Mainly contains csv files exported from Tensorboard that havebeen utilised in visualising the overal performance offered by the trained model.

### PPO_DDPG_Comparison
* Mainly contains code files targeting to solve the first research objective (Question 1)
* To investigate how agents trained using different reinforcement learning algorithms such as Proximal Policy Optimisation (PPO) and Deep Deterministic Policy Gradient (DDPG) differ in terms of performance given the same number of training timesteps.

### RGB_PPO
* Mainly contains code files targeting to solve the second research objective (Question 2)
* To conduct a research study on how training using a Convolutional Neural Network (CNN) that considers values from individual RGB channels found within the CarRacing-v0 environment affects the performance of the agent.

## Internal
### checkpoint_models
* Containing models that have been saved halfway through the full training process of a model. The reason for saving these checkpoints is to allow better tracing of appropriate timesteps that can produce a better performing model.
* 
### runs
* Containing all the Tensorboard logs of prevously conducted trainings.

### videos
* Containing short clips that facilitate a better understanding of the true performance of the model when being put into the racetrack. 

### Training
* Containing models of fully trained reinforcement learning algorithm.

# Prerequisites
### Installation of Git
* Go to this link:
[https://git-scm.com/download/win](https://git-scm.com/download/win)
* Select the version based on your machine's information.
* Extract the files followed by running of the installer.


### Installation of Anaconda
Installation of Anaconda can be accessible through this link :
https://docs.anaconda.com/anaconda/install/windows/


# Running of Source Code
* After having all the prerequisites done, you are now ready to run the cloned source code.
* Go to Anaconda Navigator and head to the environment tab on the most left part of the window.
* Search for the environment you have imported and click onto the start icon beside the enviroment to boot up the environment.
* Simply head to Home tab and search for PyCharm.
* Select 'Launch' to have PyCharm booted up.
* Within PyCharm, head to file and select 'Open''.
* Click on the files you would like to access.

# Python Version
The project within this repository utilises Python 3.7

# Attribute
> All files included inside the lib folder are written in-house.
### List of packages (Standard Libraries) that has been included into the project are as follow:
  * cudatoolkit: 3.4.8
  * gym: 0.21.0
  * keras: 2.4.3
  * matplotlib: 3.5.1
  * numpy: 1.21.5
  * opencv: 4.5.5
  * pandas: 1.3.5
  * pygame: 2.1.2
  * pyglet: 1.5.16
  * pytorch: 1.11.0
  * swig: 4.0.2
  * tensorboard: 2.8.0
  * tensorflow: 2.3.0

