from hf_example_repo.activiation_grabber import ActiviationGrabber
from hf_example_repo.subject import Subject


def main() -> None:
    print("Hello World")
    subject = Subject()
    activiation_grabber = ActiviationGrabber()
    print(subject.name)
    print(activiation_grabber.generate_layer(None))


if __name__ == "__main__":
    main()
