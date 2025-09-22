import argparse
import os
import cv2
import numpy as np


class LandmarkPicker:
    '''
    Interactive tool to pick (and edit) six 2D landmarks in an image. The resulting
    landmarks are saved to a text file in the same directory as the input image.
    Important mouse and key controlls: 
        - Left click: Add a new landmark
        - Right click: Deletes last added landmark
        - Enter: Save landmarks and continue
        - 'q' or 'ESC': Quit without saving
    :param image_or_directory: Path to an image file or a directory containing multi-view images
    '''
    def __init__(self, image_or_directory):
        # Check if image or directory. If directory with multi-view images, pick frontal-facing image.
        if os.path.isdir(image_or_directory):
            image_files = [f for f in os.listdir(image_or_directory) if f.lower().endswith(('.jpg', '.png'))]
            
            if not image_files:
                raise ValueError(f'No image files found in directory: {image_or_directory}.')
        
            image_files.sort()
            mid_index = len(image_files) // 2
            self.image_path = os.path.join(image_or_directory, image_files[mid_index])
        elif os.path.isfile(image_or_directory):
            self.image_path = image_or_directory
        else:
            raise ValueError(f'Invalid path: {image_or_directory}. \
                             Does neither contain valid image files nor is it an image.')
  
        camera_id = os.path.basename(self.image_path).split('.')[0]
        self.output_file = os.path.join(os.path.dirname(os.path.dirname(self.image_path)),
                                        f'{camera_id}_landmarks.txt')

        self.original_image = cv2.imread(self.image_path)
        self.display_image = self.original_image.copy()
        self.landmarks = []
        
        # Editing state.
        self.selected_landmark_index = -1
        self.is_dragging = False
        self.hover_landmark_index = -1  # Track which landmark we're hovering over
        
        # Landmark order for guidance.
        self.landmark_order = [
            'Sternal notch',
            'Belly button',
            'Left nipple (from the patient\'s perspective; so it\'s actually *right* from your perspective!)',
            'Right nipple (from the patient\'s perspective; so it\'s actually *left* from your perspective!)',
            'Left coracoid process (from the patient\'s perspective; so it\'s actually *right* from your perspective!)',
            'Right coracoid process (from the patient\'s perspective; so it\'s actually *left* from your perspective!)'
        ]
  
        # Display settings.
        self.window_name = '2D Landmark Picker'
        self.point_radius = 3
     
        # Scale image if too large.
        self.scale_factor = 1.0
        max_dimension = 1200
        height, width = self.original_image.shape[:2]
        if max(height, width) > max_dimension:
            self.scale_factor = max_dimension / max(height, width)
            self.display_image = cv2.resize(self.display_image,
                                            (int(width * self.scale_factor), int(height * self.scale_factor)))

    def find_nearest_landmark(self, x, y):
        if not self.landmarks:
            return -1
        
        orig_x = x / self.scale_factor
        orig_y = y / self.scale_factor
        
        min_distance, nearest_index = float('inf'), -1 
        for i, (lx, ly) in enumerate(self.landmarks):
            distance = np.sqrt((orig_x - lx) ** 2 + (orig_y - ly) ** 2)

            if distance < 15 / self.scale_factor and distance < min_distance:
                min_distance, nearest_index = distance, i

        return nearest_index

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near an existing landmark; if so, start editing it.
            nearest_idx = self.find_nearest_landmark(x, y)

            if nearest_idx != -1:
                # We're editing an existing landmark.
                self.selected_landmark_index = nearest_idx
                self.is_dragging = True
            else:
                # We're not editing. Add new landmark if we haven't reached the limit.
                if len(self.landmarks) < 6:
                    self.landmarks.append((int(x / self.scale_factor), int(y / self.scale_factor)))
         
            self.update_display()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging and self.selected_landmark_index != -1:
                # Update landmark position when dragging.
                self.landmarks[self.selected_landmark_index] = (int(x / self.scale_factor), int(y / self.scale_factor))
                self.update_display()
            else:
                # Check if hovering over a landmark for visual feedback.
                nearest_idx = self.find_nearest_landmark(x, y)
                if nearest_idx != self.hover_landmark_index:
                    self.hover_landmark_index = nearest_idx
                    self.update_display()
        elif event == cv2.EVENT_LBUTTONUP:
            # We're done dragging.
            if self.is_dragging and self.selected_landmark_index != -1:
                self.landmarks[self.selected_landmark_index] = (int(x / self.scale_factor), int(y / self.scale_factor))
                self.selected_landmark_index = -1
                self.is_dragging = False
                self.update_display()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last landmark.
            if self.landmarks:
                self.landmarks.pop()
                self.update_display()

    def update_display(self):
        self.display_image = self.original_image.copy()

        if self.scale_factor != 1.0:
            height, width = self.original_image.shape[:2]
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            self.display_image = cv2.resize(self.display_image, (new_width, new_height))
        
        # Draw selected landmarks.
        for i, (x, y) in enumerate(self.landmarks):
            display_x = int(x * self.scale_factor)
            display_y = int(y * self.scale_factor)
            
            # Highlight the selected landmark being edited.
            if i == self.selected_landmark_index and self.is_dragging:
                color = (0, 0, 255)  # Red for selected landmark.
                radius = self.point_radius + 2
            elif i == self.hover_landmark_index:
                color = (0, 0, 255)  # Red for hovered landmark (indicating it can be dragged).
                radius = self.point_radius + 1
            else:
                color = (0, 255, 0)  # Green for normal landmarks.
                radius = self.point_radius
            
            cv2.circle(self.display_image, (display_x, display_y), radius, color, -1)
            
            # Add an extra ring around hovered landmarks to make it more obvious
            if i == self.hover_landmark_index and not self.is_dragging:
                cv2.circle(self.display_image, (display_x, display_y), radius + 3, (0, 255, 255), 2)
            
            text = str(i + 1)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = display_x - text_size[0] // 2
            text_y = display_y - radius - 5
            
            cv2.rectangle(self.display_image, 
                         (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         (0, 0, 0), -1)
            
            cv2.putText(self.display_image, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add overlay text showing next landmark to pick.
        self.draw_next_landmark_overlay()

        cv2.imshow(self.window_name, self.display_image)
   
    def draw_next_landmark_overlay(self):
        if len(self.landmarks) < len(self.landmark_order):
            next_landmark = self.landmark_order[len(self.landmarks)]
            text = f'[{len(self.landmarks)}/6] Next landmark: {next_landmark}.'
        else:
            text = '[6/6] All landmarks placed. Press Enter to save and continue.'
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Position the overlay in the top-left corner with some padding.
        padding = 10
        box_x = padding
        box_y = padding
        box_width = text_size[0] + 2 * padding
        box_height = text_size[1] + 2 * padding
        
        cv2.rectangle(self.display_image, 
                     (box_x, box_y), 
                     (box_x + box_width, box_y + box_height), 
                     (128, 128, 128), -1)
        
        text_x = box_x + padding
        text_y = box_y + padding + text_size[1]

        cv2.putText(self.display_image, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def save(self):
        if not self.landmarks:
            return False
        
        np.savetxt(self.output_file, self.landmarks, fmt = '%d', delimiter = ',')
        return True

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27: # Q or ESC
                print('Quitting without saving...'); break
            elif key == 13: # Enter
                if len(self.landmarks) < 6:
                    print('Please pick exactly six landmarks.')
                elif self.save():
                    print('Saved landmarks and quitting...'); break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('image_or_directory',
                           help = 'Path to the input image or a directory containing multi-view images.')

    args, _ = argparser.parse_known_args()

    picker = LandmarkPicker(args.image_or_directory)
    picker.run()