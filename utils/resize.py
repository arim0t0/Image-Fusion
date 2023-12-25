import cv2
import numpy as np

season1 = {
    "input_pts": np.float32([[920, 6000], [5620,3115], [4165, 8490], [7655, 5025]]),
    "output_pts": np.float32([[5850, 10300], [15390, 4420], [12440, 15450], [19560, 8310]])
}

season2 = {
    "input_pts": np.float32([[2730, 1280], [1830, 3890], [4020, 2050], [2820, 4430]]),
    "output_pts": np.float32([[6890, 6450], [5480, 10490], [8910, 7640], [7010, 11390]])
}

# season3 = {
#     "input_pts": np.float32([[2730, 1280], [1830, 3890], [4020, 2050], [2820, 4430]]),
#     "output_pts": np.float32([[6930, 6420], [5460, 10490], [8910, 7640], [7020, 11370]])
# }

list_season = [1, 2, 3]

class Resize:
    def __init__(self, season, coordinates):
        for i in list_season:
            if season == i:
                self.season = "season" + str(i)
                self.season = eval(self.season)
                break
        self.input_pts = np.float32(self.season["input_pts"])
        self.output_pts = np.float32(self.season["output_pts"])

        self.coordinates = coordinates             # coordinates = [top:bottom, left:right]
        self.coordinates = self.coordinates[0], self.coordinates[1], self.coordinates[2], self.coordinates[3]

    def resize_rgb(self, main_image):
        self.original_height = main_image.shape[0]
        self.original_width = main_image.shape[1]
        self.main_image = main_image
        self.main_image = self.main_image[self.coordinates[0]:self.coordinates[1], self.coordinates[2]:self.coordinates[3]]
        return self.main_image
    
    # def resize_ndvi(self, image):
    #     image = cv2.warpPerspective(src=np.array(image),M=M, dsize=(26488,23844))
    #     image = image[4420:15390,5850:19560]
    #     return image
    
    def resize_ms(self, ms_image):
        self.ms_image = ms_image
        M = cv2.getPerspectiveTransform(self.input_pts, self.output_pts)
        self.ms_image = cv2.warpPerspective(src=np.array(self.ms_image),M=M, dsize=(self.original_width, self.original_height))
        print(self.ms_image.shape)
        self.ms_image = self.ms_image[self.coordinates[0]:self.coordinates[1], self.coordinates[2]:self.coordinates[3]]
        return self.ms_image