# coding:utf-8
import cv2
import random

def CaptureVideoFromCamera():
    # 获取视频流操作对象cap
    cap = cv2.VideoCapture(0)
    # 获取cap的视频帧
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps: ", fps)
    # 获取每帧大小
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("size: ", size)

    # 定义编码格式 mpge-4
    # 此处 fourcc 在MAC上有效
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', '2')
    # 定义视频文件输入对象，第三个参数是镜头快慢的（帧），10为正常，小于10为慢镜头
    out = cv2.VideoWriter('data/video stream/camera stream.avi', fourcc, 10, size)

    # 获取视频流打开状态
    if cap.isOpened():
        rval, frame = cap.read()
        print("Normal open")
    else:
        rval = False
        print("Nonnormal poen")

    tot = 1
    cut = 1
    ran = random.randint(0, 30)

    # 循环使用read()打开视频帧
    while rval:
        rval, frame = cap.read()
        cv2.imshow("stream", frame)

        # 任选5帧保存为图片
        if tot % ran == 0 and cut < 6:
            cv2.imwrite("data/video stream/" + "camera_cut_" + str(cut) + ".jpg", frame)
            cut += 1
            print("save successful", cut)
        tot += 1
        print("tot = ", tot)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or rval == False:
            print("done")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def SaveVideoFromFile():
    cap = cv2.VideoCapture("data/video stream/test.avi")
    # 获取cap的视频帧
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps: ", fps)
    # 获取每帧大小
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("size: ", size)

    # 定义编解码器 创建视频保存对象
    # fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')
    # 定义视频文件输入对象，第三个参数是镜头快慢的（帧），10为正常，小于10为慢镜头
    # out = cv2.VideoWriter('data/video stream/output.avi', fourcc, 10, size)

    # 获取视频流打开状态
    if cap.isOpened():
        rval, frame = cap.read()
        print("Normal open")
    else:
        rval = False
        print("Nonnormal poen")

    tot = 1
    cut = 1
    ran = random.randint(9, 70)

    # 循环使用read()打开视频帧
    while rval:
        rval, frame = cap.read()
        cv2.imshow("stream", frame)
        # out.write(frame)

        # 任选5帧保存为图片
        if tot % ran == 0 and cut < 6:
            cv2.imwrite("data/video stream/" + "avi_cut_" + str(cut) + ".jpg", frame)
            cut += 1
            print("save successful", cut)
        tot += 1
        print("tot = ", tot)

        if cv2.waitKey(1) & 0xFF == ord('q') or rval == False:
            print("done")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("1. Capture Video From Camera" + "\n"
          "2. Save Video From File")
    get = input()

    if get == "1":
        CaptureVideoFromCamera()
    elif get == "2":
        SaveVideoFromFile()
    else:
        print("Invalid input")


