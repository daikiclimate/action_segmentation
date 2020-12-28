#demo

```
mkdir data
mkdir data/demo
cd src/demo
```

and read README.md in demo.
Learned weight is in models/vgg.

You need to put your Splatoon movie in data/videos/

Only image_model is valid now

#training

training code is in src/models/
image_model is a model for each frame
video_model is a model for video which considers context of previous and suffix flames. video_model is not Implemented
