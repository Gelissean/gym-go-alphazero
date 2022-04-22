import csv
import os


class data_exporter():
    def __init__(self, path):
        done = False
        temppath = path
        while not done:
            if os.path.exists(temppath):
                temppath = temppath[0:-3] + '1' + '.csv'
            else:
                done = True
        self.f = open(path, 'w')
        self.writer = csv.writer(self.f)

    def __del__(self):
        if self.f is not None:
            self.f.close()

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.f.close()
        self.f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f is not None:
            self.close()

def main():
    with data_exporter("subor.csv") as exporter:
        exporter.write(["udaj", "druhy","treti"])


if __name__ == '__main__':
    main()