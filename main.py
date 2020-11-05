from app.main import CameraApp

if __name__ == '__main__':

    app = CameraApp()
    app.detect_platform()
    app.run()
