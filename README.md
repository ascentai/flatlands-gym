# Flatlands Driving Simulator for OpenAI Gym

## Ascent Robotics

Flatlands is a simple gym compatible simulator for a car driving around a track.

To use it, clone this repo to your computer with:
```
git clone git@github.com:ascentai/flatlands-gym.git && cd flatlands-gym
```

Then install the dependencies with:
```
pip install -e .
```

To create a gym environment you need to import `gym` and then import `flatlands` inside python
```python
import gym
import flatlands

env = gym.make("Flatlands-v0")
```

For a more in depth example, see [demo_flatlands.py](demo_flatlands.py) which drives that car based on the steering angle compared to upcoming points.
