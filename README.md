# Flatlands Driving Simulator for OpenAI Gym

## Ascent Robotics

Flatlands is a simple gym compatible simulator for a car driving around a track. It is tested with Python3, and Windows/ Ubuntu.

### Automatic Installation
You can install flatlands from PyPi with:
```shell-script
pip install flatlands
```

### Manual Installation
- You can install manually from source by cloning this repo to your computer with:
```shell-script
git clone git@github.com:ascentai/flatlands-gym.git && cd flatlands-gym
```

- Then install it with:
```shell-script
pip install -e .
```
> Flatlands uses `pygame` for visualization, so if this doesn't work, you may need to install the SDL library on your system as well.


### Usage
- To create a gym environment you need to `import gym` and then `import flatlands` from inside python. Importing flatlands is enough to register it with the gym registry.
```python
import gym
import flatlands

env = gym.make("Flatlands-v0")
```

For a more in depth example, see [demo_flatlands.py](demo_flatlands.py) which drives that car based on the steering angle compared to upcoming points.

The [Gym documentation](https://gym.openai.com/docs/#observations) explains more about interacting with an environment
