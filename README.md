# Flatlands Driving Simulator for OpenAI Gym

## Ascent Robotics

Flatlands is a simple gym compatible simulator for a car driving around a track. It is tested with Python3, and Windows/ Ubuntu.

- To use it, clone this repo to your computer with:
```shell-script
git clone git@github.com:ascentai/flatlands-gym.git && cd flatlands-gym
```

- Then install the dependencies with:
```shell-script
pip install -e .
```
> Flatlands uses `pygame` for visualization, so if this doesn't work, you may need to install the SDL library on your system as well.

- To create a gym environment you need to import `gym` and then import `flatlands` inside python
```python
import gym
import flatlands

env = gym.make("Flatlands-v0")
```

For a more in depth example, see [demo_flatlands.py](demo_flatlands.py) which drives that car based on the steering angle compared to upcoming points.
