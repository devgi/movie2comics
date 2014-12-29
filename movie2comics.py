import os
import collections

import logbook
import peewee
import cv2
import pysrt


logger = logbook.Logger('Movie2Comics')

# # Paths

VIDEO = r"E:\DOWNLOADS\Happy Endings Season 1 Complete 720p\Happy.Endings.S01E01.720p.HDTV.x264.mkv"
SUBTITLES = r"E:\DOWNLOADS\Happy Endings Season 1 Complete 720p\Happy.Endings.S01E01.720p.HDTV.x264.srt"

WORKSPACE = r'e:\workspace\movie2comics'
DB_PATH = os.path.join(WORKSPACE, 'm2c.db')

# # DB UTILS

db = peewee.SqliteDatabase(None)


class BaseModel(peewee.Model):
    class Meta:
        database = db


class Track(BaseModel):
    character = peewee.CharField(max_length=255, null=True)


class Face(BaseModel):
    """
       +(x,y) ------ (x+width)
       |
       |
       |
       +(y + height) ----
    """
    frame_index = peewee.IntegerField()

    x = peewee.IntegerField()
    y = peewee.IntegerField()
    width = peewee.IntegerField()
    height = peewee.IntegerField()


    @property
    def left_top_point(self):
        return (self.x, self.y)

    @property
    def right_top_point(self):
        return (self.x, self.x + self.width)

    @property
    def left_bottom_point(self):
        return (self.y, self.y + self.height)

    @property
    def right_bottom_point(self):
        return (self.x + self.width, self.y + self.height)


    @property
    def size(self):
        return self.width * self.height

    track = peewee.ForeignKeyField(Track, related_name='faces', null=True)


def connect(path):
    logger.info("Connect to database at: %s" % path)
    db.init(path)
    db.create_tables([Face, Track], safe=True)
    return db


## OPENCV UTILS

def display(read_result):
    if len(read_result) == 2:
        assert len(read_result) == 2 and read_result[0] == True
        imgae = read_result[1]
    else:
        image = read_result
    cv2.imshow('img', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def open_video(path):
    video = cv2.VideoCapture()
    video.open(path)
    assert video.isOpened(), "failled to open video!"
    return video


def open_video_writer(video, path, color=True, width=None, height=None, frame_size_factor=1):
    four_cc = cv2.cv.CV_FOURCC(*'XVID')
    fps = int(video.get(cv2.cv.CV_CAP_PROP_FPS))
    frame_size = int(video.get(cv2.cv.CV_CAP_PROP_FPS))
    frame_width = width or int(
        video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH) * frame_size_factor)
    frame_height = height or int(
        video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) * frame_size_factor)
    res_video = cv2.VideoWriter()
    print res_video.open(path, four_cc, fps, (frame_width, frame_height), color)

    assert res_video.isOpened()
    return res_video


def jump_to_frame_index(video, frame_index):
    res = video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_index)
    assert res, 'Failed?'


def get_current_frame_index(video):
    return video.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)


def jump_to_time(video, frame_index):
    res = video.set(cv2.cv.CV_CAP_PROP_POS_MSEC, frame_index)
    assert res, 'Failed?'


def get_current_time(video):
    return video.get(cv2.cv.CV_CAP_PROP_POS_MSEC)


def draw_face_on_image(image, face):
    cv2.rectangle(image,
                  face.left_top_point,
                  face.right_bottom_point,
                  (0, 255, 0),
                  10)

def crop_face(image, face):
    return image[face.y:face.y + face.height, face.x:face.x + face.width,]


## MOVIE TO COMICS

def subtitle_time_to_ms(sub_time):
    MS_IN_HOUR = 3600000
    MS_IN_MINUTE = 60000
    MS_IN_SEC = 1000
    MS_IN_MS = 1
    return (MS_IN_MS * sub_time.milliseconds +
            MS_IN_SEC * sub_time.seconds +
            MS_IN_MINUTE * sub_time.minutes +
            MS_IN_HOUR * sub_time.hours)


def iterate_subtitles(video, subtitles):
    for subtitle in subtitles:
        start_time_in_ms = subtitle_time_to_ms(subtitle.start)
        end_time_in_ms = subtitle_time_to_ms(subtitle.end)

        logger.debug('Process subtitle: "%s" (index %d, '
                     'start time: %d , end time: %d' % (subtitle.text,
                                                        subtitle.index,
                                                        start_time_in_ms,
                                                        end_time_in_ms))


        logger.debug("Jump to subtitle start time.")
        jump_to_time(video, start_time_in_ms)
        frame_time = -1
        while frame_time < end_time_in_ms:
            _, frame = video.read()
            yield frame, subtitle
            frame_time = get_current_time(video)


class FaceDetector(object):
    RESIZE_FACTOR = .5

    def __init__(self):
        self._face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')

    def find_faces_in_image(self, image):
        resized_image = cv2.resize(image, (0, 0),
                                   fx=self.RESIZE_FACTOR,
                                   fy=self.RESIZE_FACTOR)
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        #print "Found {0} faces!".format(len(faces))

        for (x, y, w, h) in faces:
            scale_factor = (1 / self.RESIZE_FACTOR)
            x *= scale_factor
            y *= scale_factor
            w *= scale_factor
            h *= scale_factor
            yield (x, y, w, h)

    def find_all_faces(self, video, subtitles):
        for (frame, _) in iterate_subtitles(video, subtitles):
            frame_index = get_current_frame_index(video)
            logger.debug("Process frame %d" % frame_index)
            for (x, y, w, h) in self.find_faces_in_image(frame):
                logger.debug("Found face at (x=%d,y=%d)" % (x, y))
                Face.create(frame_index=frame_index-1,
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            track=None)



# def face_overlap_size(face1, face2):
#     top_left = max(face1.left_top_point, face2.left_top_point)
#     top_right = min(face1.right_top_point, face2.right_top_point)
#     bottom_left = max(face1.left_bottom_point, face2.left_bottom_point)
#     bottom_right = min(face1.right_bottom_point, face2.right_bottom_point)
#
#     assert (top_left[0] == top_right[0] and
#             top_left[1] == bottom_left[1] and
#             bottom_left[0] == bottom_right[0] and
#             top_right[1] == bottom_right[1])
#
#     width = top_left[0] - top_right[0]
#     height = top_right[1] - bottom_right[1]
#
#     assert width >= 0 and height >= 0
#     return width * height

def face_overlap_size(face1, face2):
    top_left_x = max(face1.x, face2.x)
    top_left_y = max(face1.y, face2.y)
    bottom_left_x = min(face1.x + face1.width, face2.x + face2.width)
    bottom_left_y = min(face1.y + face1.height, face2.y + face2.height)

    width = max(0, bottom_left_x - top_left_x)
    height = max(0, bottom_left_y - top_left_y)

    assert width >= 0 and height >= 0
    return width * height


def connect_tracks():
    """
    For each face detected, look at the faces in the previous frame and
    try to find one that overlaps.
    """
    OVERLAP_THRESHOLD = 0.8

    for face in Face.select().order_by(Face.frame_index):
        prev_frame_index = face.frame_index - 1

        track = None
        for face_in_prev_frame in Face.filter(frame_index = prev_frame_index):
            overlap_size = face_overlap_size(face_in_prev_frame, face)
            size_ratio = float(overlap_size) / face.size
            assert 0 <= size_ratio <= 1
            if size_ratio > OVERLAP_THRESHOLD:
                assert face_in_prev_frame.track
                track = face_in_prev_frame.track
                logger.debug("For face %d: found face with %f "
                             "overlap on track %d" % (face.id,
                                                      size_ratio,
                                                      track.id))

        if track is None:
            track = Track.create(character="")
            logger.debug('Track not found for face (%d) new '
                         'track created (%d)' % (face.id, track.id))

        face.track = track
        face.save()

def prone_tracks():
    """
    Delete tracks with less than PRONE_THRESHOLD faces.
    """
    PRONE_THRESHOLD = 4
    for track in list(Track.select()):
        if track.faces.count() < PRONE_THRESHOLD:
            logger.debug("Track %d have %d faces, remove it" % (track.id,
                                                                track.faces.count()))

            for face in list(track.faces):
                logger.debug("Delete face %d" % face.id)
                face.delete_instance()

            logger.debug("Delete track %d" % track.id)
            track.delete_instance()




##

def track_to_video(track, video, path_to_track_video):
    min_w = min(f.width for f in track.faces)
    min_h = min(f.height for f in track.faces)

    vw = open_video_writer(video, path_to_track_video, width=min_w, height=min_h)
    for f in track.faces:
        jump_to_frame_index(video, f.frame_index)
        _, img = video.read()
        face_img = crop_face(img, f)
        new_fimg = cv2.resize(face_img, (min_w, min_h))
        vw.write(new_fimg)
    vw.close()



