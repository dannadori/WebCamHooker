# WebCamHooker
- Web link(Japanese): https://cloud.flect.co.jp/entry/2020/03/31/162537
- Previous: see README2.md

This time, I’ve expanded it a little more and experimented with converting an image into an animation-style display, so here it is.

<p align="center">
<img src="./doc/fps_test2.gif" width="800" />
</p>


#### Assumptions.
Take a look at previous article(see README.md) and set up v4l2loopback, etc.

#### Webcam hook extensions
As mentioned above, it seems that the image should be close to the subject (the face of the person) (≒the face occupies most of the screen). This time, I tried to identify the location of the face by using the face detection function, which I introduced in the previous articles, and then I cut out the location of the face and applied the transformation by UGATIT.

First of all, please clone the script and install the necessary modules from the following repository as in the previous one.

```
$ git clone https://github.com/dannadori/WebCamHooker.git
$ cd WebCamHooker/
$ pip3 install -r requirements.txt
```


Next, get a pre-trained model from UGATIT.
Here is the hash value (md5sum) of the model that runs normally. (Probably because this is the biggest stumbling block.)
```
$ find . -type f |xargs -I{} md5sum {}
43a47eb34ad056427457b1f8452e3f79 . /UGATIT.model-1000000.data-00000-of-00001
388e18fe2d6cedab8b1dbaefdddab4da . /UGATIT.model-1000000.meta
a0835353525ecf78c4b6b33b0b2ab2b75c . /UGATIT.model-1000000.index
f8c38782b22e3c4c61d4937316cd3493 . /checkpoint
```
These files are stored in `UGATIT/checkpoint` in the folder you cloned from the above git. If it looks like this, it’s OK.
```
$ ls UGATIT/checkpoint/ -1
UGATIT.model-1000000.data-00000-of-00001
UGATIT.model-1000000.index
UGATIT.model-1000000.meta
checkpoint
```


The execution is as follows. One option has been added.

- The input_video_num should be the actual webcam device number. For /dev/video0, enter a trailing 0.
- The output_video_dev must be the device file of the virtual webcam device.
- anime_mode should be true.
In addition, please use ctrl+c to terminate.


```
$ python3 webcamhooker.py --input_video_num 0 --output_video_dev /dev/video2 --anime_mode True
```

When the above command is executed, ffmpeg starts to run and the video is delivered to the virtual camera device.

#### Let's do a video conference!
As before, when you have a video conference, you should see something called "dummy~" in the list of video devices, so select it.
