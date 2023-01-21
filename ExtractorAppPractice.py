import json
import os
import shutil
import tkinter as tk
from collections import namedtuple
from datetime import date
from tkinter.filedialog import askopenfile
from tkinter.ttk import *

import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image, ImageTk
from pdf2image import convert_from_path
from tkPDFViewer import tkPDFViewer as pdf

t = 0
list_of_files = []
drop_down_menu_width = 0
path_of_pdf = ""
v2 = None

pytesseract.pytesseract.tesseract_cmd = r'Add_ons/Tesseract-OCR/tesseract.exe'

poppler_path = r'Add_ons\poppler-21.11.0\Library\bin'

x1 = None
y1 = None
x2 = None
y2 = None


class GetCoords(tk.Toplevel):
    def __init__(self, master, image_path, size):
        self.image_path = image_path
        self.size = size
        tk.Toplevel.__init__(self, master)
        self.grab_set()
        self.x = self.y = 0
        self.canvas = tk.Canvas(self, height=1000, width=1000, cursor="cross")
        self.canvas.pack(expand=True)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None

        self.start_x = None
        self.start_y = None
        self._draw_image()


    def _draw_image(self):
        self.im = Image.open(self.image_path)
        self.im = self.im.resize((1000, 1000), Image.ANTIALIAS)
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0,0,anchor="nw", image=self.tk_im)

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline= 'red', width = 4)

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        # Remember that this data is not vector coordinates, but x and y coordinate plus height and width
        print(self.start_x, self.start_y, event.x, event.y, "heyy")
        factorx = self.size[0] / 1000
        factory = self.size[1] / 1000
        global x1, x2, y1, y2
        x1 = int(self.start_x * factorx)
        y1 = int(self.start_y * factory)
        x2 = int((event.x - self.start_x) * factorx)
        y2 = int((event.y - self.start_y) * factory)
        print(self.size, factorx, factory)
        print(x1,y1,x2,y2)
        self.destroy()
        pass


def retrieve_template_data(template_name, fields_names, return_type):
    if return_type == "fields":
        # with open('Resources\\ocr_location_data.json', 'r') as f:
        with open('Resources\\field_coordinates.json', 'r') as f:
            Locations = []
            template_list = json.load(f)
            template_data = template_list[template_name]
            for i in fields_names:
                dict_info = template_data[i]
                field_data = namedtuple("OCRLocation", ["id", "bbox", "page_num", "filter_keywords"])
                data = field_data(i, tuple(dict_info["grid_coordinates"]), dict_info["page_number"],
                                  dict_info["key_words"])
                print(data)
                Locations.append(data)
            return Locations
    else:
        with open('Resources\\field_coordinates.json', 'r') as f:
            template_list = json.load(f)
            template_data = template_list[template_name]
            return template_data.keys()


class ImageEx:
    def __init__(self, name, pdf_path, image_path):
        self.name = name
        self.pdf_path = pdf_path
        self.image_path = image_path

    def save_images(self):
        if ".pdf" in self.pdf_path:
            images = convert_from_path(str(self.pdf_path), poppler_path=poppler_path)
        else:
            images = convert_from_path(str(self.pdf_path + self.name + '.pdf'), poppler_path=poppler_path)
        list_of_image_paths = []
        for i in range(len(images)):
            path = self.image_path + self.name + str(i + 1) + '.PNG'
            images[i].save(path)
            list_of_image_paths.append(self.name + str(i + 1) + '.PNG')
        return list_of_image_paths


def align_images(image, template, maxFeatures=500, keepPercent=0.2,
                 debug=False):
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)
    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt
    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    # return the aligned image
    return aligned


class GetInfo:
    def __init__(self, pdf_path, name_of_template, fields):
        self.pdf_path = pdf_path
        self.name_of_template = name_of_template
        self.fields = fields

    def get_image_paths_and_fields_and_field_groups(self):
        imgex = ImageEx(self.name_of_template, self.pdf_path, os.getcwd() + '\\Temporary\\Images\\')
        # imgex.download_pdf()
        image_paths = imgex.save_images()
        # gives use back selected fields and their coordinate information
        field_coordinate_info = retrieve_template_data(self.name_of_template, self.fields, "fields")
        return image_paths, field_coordinate_info


class ExtractData(GetInfo):

    @staticmethod
    def comp_vision_extraction(image_path, template_path, field_coordinate_info):

        def cleanup_text(data_text):
            # strip out non-ASCII text so we can draw the text on the image using OpenCV
            return "".join([c if ord(c) < 128 else "" for c in data_text]).strip()

        #print("[INFO] loading images...")
        #print(image_path)
        # getting cv2 object of image and template
        image = cv2.imread(image_path)
        template = cv2.imread(template_path)
        # align the images
        #print("[INFO] aligning images...")
        aligned = align_images(image, template)

        #print("[INFO] OCR'ing document...")
        parsingResults = []
        # loop over the locations of the document we are going to OCR
        for loc in field_coordinate_info:
            # extract the OCR ROI from the aligned image
            (x, y, w, h) = loc.bbox
            roi = aligned[y:y + h, x:x + w]
            # OCR the ROI using Tesseract
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(rgb)
            # break the text into lines and loop over them
            for line in text.split("\n"):
                # if the line is empty, ignore it
                if len(line) == 0:
                    pass
                else:
                    # convert the line to lowercase and then check to see if the
                    # line contains any of the filter keywords (these keywords
                    # are part of the *form itself* and should be ignored)
                    lower = line.lower()
                    count = sum([lower.count(x) for x in loc.filter_keywords])
                    # if the count is zero then we know we are *not* examining a
                    # text field that is part of the document itself (ex., info,
                    # on the field, an example, help text, etc.)
                    if count == len(loc.filter_keywords):
                        # update our parsing results dictionary with the OCR'd
                        # text if the line is *not* empty
                        parsingResults.append((loc, line))
                    # initialize a dictionary to store our final OCR results`
        results = {}
        # loop over the results of parsing the document
        for (loc, line) in parsingResults:
            # grab any existing OCR result for the current ID of the document
            r = results.get(loc.id, None)
            # if the result is None, initialize it using the text and location
            # namedtuple (converting it to a dictionary as namedtuples are not
            # hashable
            if r is None:
                #print("results None", loc.id, r)
                results[loc.id] = (line, loc._asdict())
            # otherwise, there exists an OCR result for the current area of the
            # document, so we should append our existing line
            else:
                # unpack the existing OCR result and append the line to the
                # existing text
                (existingText, loc) = r
                text = "{}\n{}".format(existingText, line)
                #print("line: ", line)
                # update our results dictionary
                results[loc["id"]] = (text, loc)
        # loop over the results
        return_list = []
        for (locID, result) in results.items():
            # unpack the result tuple
            (text, loc) = result
            text = cleanup_text(text)
            data = [loc["id"], text]
            return_list.append(data)
        return return_list

    def return_field_data(self):
        # image_paths is a queue of images paths from 1 page to last page

        image_paths, field_coordinate_info = super().get_image_paths_and_fields_and_field_groups()
        template_directory = 'Resources/Templates/' + self.name_of_template
        template_path = os.listdir(template_directory)



        # gets page count for the doc
        sorted_object = sorted(field_coordinate_info, key=lambda x: x.page_num)
        page_count = sorted_object[-1].page_num
        extracted_data = []
        list_of_obj_index = 0
        current_page = 1
        size = len(sorted_object) - 1
        # looping through list of OCRLocation class objects to find only objects that have same page
        # the two while loops speed up the process by making sure that we only loop through the object list
        # starting at the index that contains the relevant page number
        while current_page <= page_count:
            fields_in_page = []
            while list_of_obj_index <= size and sorted_object[list_of_obj_index].page_num == current_page:
                fields_in_page.append(sorted_object[list_of_obj_index])
                list_of_obj_index += 1
            img_path = image_paths[current_page - 1]
            temp_path = template_path[current_page - 1]
            data = self.comp_vision_extraction('Temporary//Images//' + img_path, template_directory + '/' + temp_path,
                                               fields_in_page)

            extracted_data = extracted_data + data
            current_page += 1
        return extracted_data


def get_all_templates():
    global list_of_files
    global drop_down_menu_width

    list_of_files = os.listdir(os.getcwd() + '\\Resources\\Templates')
    drop_down_menu_width = len(max(list_of_files, key=len))


def reset(i, container, self):
    val = i(container, self)
    return val


def display_pdf(previous_node):
    global v2
    global path_of_pdf
    if v2:
        # destroying old instance of v2. This is necessary!!!
        v2.destroy()
    print("print pdf path", path_of_pdf)
    pdf_dis = tk.Toplevel(previous_node)
    pdf_dis.geometry("550x750")
    v1 = pdf.ShowPdf()
    # clear the stored image list
    v1.img_object_li.clear()
    v2 = v1.pdf_view(pdf_dis, pdf_location=path_of_pdf, width=50, height=100)
    v2.pack()



def open_popup(results, size, title, should_display_pdf):
    try:
        global app
        popup = tk.Toplevel(app)
        popup.geometry(size)
        popup.title(title)
        if type(results) is list:
            row = 1
            for x in results:
                Label(popup,  text=x[0] + ": ", anchor="w", font=('Helvetica 11 bold'), foreground="red").grid(column=1,
                                                                                                              row=row,
                                                                                                              padx=10,
                                                                                                              pady=10)
                Label(popup, text=x[1], font=('Helvetica 11 bold')).grid(column=2, row=row, padx=10, pady=10)
                row += 1
        else:
            Label(popup, text=results, font=('Helvetica 11 bold'), foreground="red").grid(pady=70, padx=70)
        if should_display_pdf is not None:
            display_pdf(popup)
    except Exception as e:
        with open('App_Log.txt', 'w') as f:
            f.write('%s' % getattr(e, 'message', repr(e)))
            f.close()



def on_start():
    list_of_imgs = os.listdir(os.getcwd() + '\\Temporary\\Images\\')
    list_of_pdfs = os.listdir(os.getcwd() + '\\Temporary\\PDFs\\')
    if list_of_imgs:
        for i in list_of_imgs:
            path_of_temp_img = os.getcwd() + '\\Temporary\\Images\\' + i
            os.remove(path_of_temp_img)
    elif list_of_pdfs:
        for x in list_of_pdfs:
            path_of_temp_pdfs = os.getcwd() + '\\Temporary\\Images\\' + x
            os.remove(path_of_temp_pdfs)
    else:
        print("ran")


class StartPage(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.iconbitmap(self, default="mybestplan_Jd2_icon.ico")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        global t
        menubar = tk.Menu(container)
        menubar.add_command(label="Extract Data", command=lambda: self.show_frame(Template_Selector, container))
        menubar.add_command(label="Create Template", command=lambda: self.show_frame(Template_Creator, container))
        tk.Tk.config(self, menu=menubar)
        get_all_templates()
        self.frames = {}
        for F in (Template_Selector, Template_Creator):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        get_all_templates()
        self.show_frame(Template_Selector, container)
        t += 1

    def show_frame(self, cont, container):
        global t
        frame = self.frames[cont]
        print("Value of Global", t)
        # Have to reload the frame with new data because new template might have been added
        if cont is Template_Selector and t > 0:
            get_all_templates()
            val = reset(cont, container, self)
            self.frames[cont] = val
            val.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()


class Template_Selector(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        global list_of_files
        global drop_down_menu_width

        def get_fields_for_selection(template_selected):
            global list_of_files
            list_of_fields = retrieve_template_data(template_selected, "none", "none")
            return list_of_fields

        def deselect_fields():
            for key, item in list_selections.items():
                if 'selected' in item.state():
                    item.invoke()
                else:
                    item.invoke()

        def create_field_selection(*args):
            row_num = 7
            count = 0
            column_number = 2
            for button in buttons:
                button.grid_remove()
            list_selections.clear()
            name = template.get()
            print(name)
            name = name.split(".PNG")[0]
            list_of_fields = get_fields_for_selection(str(name))

            for x in list_of_fields:
                if count % 10 == 0 and count >= 10:
                    column_number += 1
                    row_num = 7

                buttons.append(Checkbutton(self, text=x, onvalue=1, offvalue=0))
                buttons[-1].grid(column=column_number, row=row_num, sticky="w")
                buttons[-1].invoke()
                list_selections[x] = buttons[-1]
                row_num += 1
                count += 1
            button_extractor.grid(column=2, row=17, padx=10, pady=10)
            deselector_button.grid(column=3, row=17, padx=10, pady=10)
            print(list_of_fields, list_selections, name)

        def get_field_values():
            fields_to_extract = []
            for key, item in list_selections.items():
                if 'selected' in item.state():
                    fields_to_extract.append(key)
            return fields_to_extract

        '''def progress():
            progresslabel = Label(self, text="Extracting Data....", font="bold")
            progressbar = Progressbar(self, orient="horizontal", length=100, mode='indeterminate')
            progresslabel.grid(row=18, column=2)
            progressbar.grid(row=18, column=3)
            for i in range(5):
                self.update_idletasks()
                progressbar['value'] += 20
            progressbar.destroy()
            progresslabel.destroy()'''

        def select_file():
            global path_of_pdf
            path = askopenfile(mode='r', filetypes=[('PDF Files', '*pdf')])
            path_of_pdf = str(path.name)
            entry_label.config(text=path_of_pdf)
            entry_label.grid()
            print("path", path_of_pdf)

        def get_text_input():
            global path_of_pdf
            fields_to_extract = get_field_values()
            template_path: str = template.get()
            print("hey", type(path_of_pdf), path_of_pdf)
            if template_path != "" and path_of_pdf != "":
                print("Data Extraction Can Proceed")
                results = ExtractData(path_of_pdf, template_path, fields_to_extract).return_field_data()
                open_popup(results, "750x250", "Output", path_of_pdf)
            else:
                if template_path == "":
                    template_warning.grid()
                elif path_of_pdf == "":
                    file_warning.grid()

        list_selections = {}
        buttons = []

        # Title
        label_title = Label(self, text="Extract Customer Data", font="Helvetica 11 bold", anchor="center")
        label_title.grid(column=2, row=0, pady=30, padx=10)

        file_path_selector = Button(self, text='Choose PDF',
                                    command=lambda: select_file())
        file_path_selector.grid(column=2, row=2, pady=10)
        file_warning = Label(self, text="Please Select a File", font='Helvetica 9 bold', anchor='w', foreground='red')
        file_warning.grid(column=3, row=2, columnspan=2)
        file_warning.grid_remove()
        entry_label = Label(self, font='Helvetica 9 bold', anchor='w')
        entry_label.grid(column=2, row=3)
        entry_label.grid_remove()

        template = tk.StringVar()

        drop = OptionMenu(self, template, list_of_files[0], *list_of_files, command=create_field_selection)
        drop.configure(width=drop_down_menu_width)
        drop.grid(column=2, row=4, pady=10)
        template_warning = Label(self, text="Please Select a Template", font='Helvetica 9 bold', anchor='w',
                                 foreground='red')
        template_warning.grid(column=3, row=4, columnspan=2)
        template_warning.grid_remove()

        print(template.get())

        # -----
        # Button Extractor
        button_extractor = Button(self, text="Extract Data", command=get_text_input)

        # -----
        # Deselect All Button
        deselector_button = Button(self, text="Deselect All Fields", command=deselect_fields)


class Template_Creator(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)


        label_title = Label(self, text="Template Creator", font="Helvetica 14 bold", anchor="w")
        label_title.grid(row=0, column=2, pady=10)
        label_title = Label(self, text="Convert PDF to Image", font="Helvetica 9 bold", anchor="w")
        label_title.grid(row=1, column=2, pady=10)

        dict_of_fields = {}
        dict_of_meta_data = {}
        key_words = []

        class FieldData:
            def __init__(self, xcor_, ycor_, wcor_, hcor_, pnum_):
                self.xcor = self.validate_int(xcor_)
                self.ycor = self.validate_int(ycor_)
                self.wcor = self.validate_int(wcor_)
                self.hcor = self.validate_int(hcor_)
                self.pnum = self.validate_int(pnum_)

            @staticmethod
            def validate_int(x):
                if x.isnumeric() is False:
                    raise ValueError("Coordinate(s) Must Be A Number!")
                else:
                    pass

        # Function helps save the templates to the application's template folder
        # It creates a new folder in the templates with the template name and deletes the temp images.
        def create_folder(name_val):
            path_of_folder1 = os.getcwd() + '\\Resources\\Templates\\' + name_val
            print("name of new directory: ", path_of_folder1)
            print("directory name: ", name_val)
            os.makedirs(path_of_folder1, exist_ok=True)
            list_of_images = os.listdir(os.getcwd() + '\\Temporary\\Images\\')
            for i in list_of_images:
                path_of_temp_image = os.getcwd() + '\\Temporary\\Images\\' + i
                shutil.copy(path_of_temp_image, path_of_folder1)
                os.remove(path_of_temp_image)

        def open_file(new_template_name):
            if new_template_name != "":
                template_error_message.grid_remove()
                print(new_template_name)
                file_path = askopenfile(mode='r', filetypes=[('PDF Files', '*pdf')])
                if file_path is not None:
                    print(file_path.name)
                    imagex = ImageEx(name=new_template_name, pdf_path=file_path.name,
                                     image_path=os.getcwd() + '\\Temporary\\Images\\')
                    imagex.save_images()
                    path_of_folder = os.getcwd() + '\\Temporary\\Images\\'
                    # UploadSuccessful = Label(self, text='PDF Converted to Image. Please open Images in Gimp. Folder Path: ' + str(path_of_folder), foreground='green')
                    # UploadSuccessful.grid(row=3, column=1, columnspan=5, pady=10, sticky="w")
                    #instruction.grid()
                    name_label.grid()
                    Name.grid()
                    x_label.grid()
                    x_value.grid()
                    y_label.grid()
                    y_value.grid()
                    w_label.grid()
                    w_value.grid()
                    h_label.grid()
                    h_value.grid()
                    p_label.grid()
                    page_dropdown.grid()
                    key_label.grid()
                    key_word_entry.grid()
                    add_key_word_btn.grid()
                    add_field_btn.grid()
                    select_data_form_image.grid()
                    approve_selection.grid()
                    pass
            else:
                template_error_message.grid(row=2, column=4, padx=5)

        def add_field_data():
            try:
                field_info = FieldData(xcor.get(), ycor.get(), wcor.get(), hcor.get(), pnum.get())
                field_name = name_value.get()
                if field_name.isnumeric():
                    raise ValueError("Name Cannot Be A Number")
                else:
                    pass
                dict_of_fields[field_name] = field_info
                print(dict_of_fields.keys())
                val = [field_name, xcor.get(), ycor.get(), wcor.get(), hcor.get(), pnum.get(), str(key_words)]
                indx = len(dict_of_fields) - 1
                list_of_fields.insert(parent='', index=indx, iid=indx, values=tuple(val))

                dict_meta = {"grid_coordinates": [int(xcor.get()), int(ycor.get()), int(wcor.get()), int(hcor.get())],
                             "page_number": pnum.get(), "key_words": key_words}
                print(dict_meta)

                for x in [name_value, xcor, ycor, wcor, hcor, pnum]:
                    x.set("")
                dict_of_meta_data[field_name] = dict_meta
                key_words.clear()
                # we don't need to call these functions again if we already called them
                add_field_btn.grid()
                list_of_fields.grid()
                add_template_button.grid()
            except Exception as e:
                msg = str(e)
                open_popup(results=msg, size="400x200", title="Error", should_display_pdf=None)
                print(dict_of_fields)

        # This function converts the dictionary format of the fields and their metadata into json format
        def convert_to_json_formatted_string(dic, name):
            string = "  \"%s\": {\n" % name
            number_of_commas = len(dic.keys()) - 1
            for key, item in dic.items():
                string += "\t \"%s\": {" % str(key)
                size_of_keys = len(item.keys()) - 1
                for key2, item2 in item.items():
                    if size_of_keys > 0:
                        string += "\n\t  \"{key}\": {item},".format(key=key2, item=item2)
                    else:
                        string += "\n\t  \"{key}\": {item}".format(key=key2, item=item2)
                    size_of_keys -= 1
                if number_of_commas > 0:
                    string += "\n\t },\n"
                else:
                    string += "\n\t }"
                number_of_commas -= 1
            string += "\n  }"
            return string

        def log_meta_data():
            # This function basically logs the meta data for the templates in the json and also moves creates and stores
            # templates in unique folders for future use
            company_name = str(template_name_entry.get())
            added_data = convert_to_json_formatted_string(dict_of_meta_data, company_name)
            '''We are injecting the json file with a json formated string. This avoids having to taking data out of the
            json, appending to it, and then overwriting the file, which could lead to file corruption. 
            I didn't want to touch template metadata that I wasn't using'''
            with open('Resources/field_coordinates.json', 'rb+') as f:
                f.seek(-3, 2)
                data = bytes(',\n' + added_data + '\n}', 'utf-8')
                f.write(data)
                f.close()
            create_folder(company_name)
            list_of_fields.delete(*list_of_fields.get_children())
            print("dictionary of meta data from log meta data function: ", dict_of_meta_data)

        def add_key_word():
            key_words.append(str(key_word.get()))
            key_word.set('')
            print(key_words)

        def select_coords_from_pic(cl):
            list_of_images = os.listdir(os.getcwd() + '\\Temporary\\Images\\')
            #needs to default as 1
            page = int(pnum.get()) - 1
            pp = 'Temporary\\Images\\'+list_of_images[page]
            img = Image.open(pp)
            size = [img.width, img.height]
            print(img.width, img.height)
            GetCoords(cl, pp, size)

        def get_cors():
            global x1, y1, x2, y2
            xcor.set(x1)
            ycor.set(y1)
            wcor.set(x2)
            hcor.set(y2)

            print(x1, "yo")








        template_name = tk.StringVar()
        name_value = tk.StringVar()
        xcor = tk.StringVar()
        ycor = tk.StringVar()
        wcor = tk.StringVar()
        hcor = tk.StringVar()
        pnum = tk.StringVar()
        key_word = tk.StringVar()

        template_label = Label(self, text="Enter Name of Your New Template")
        template_label.grid(row=2, column=1)
        template_name_entry = Entry(self, textvariable=template_name)
        template_name_entry.grid(row=2, column=2)
        file_selector = Button(self, text='Choose PDF of Template',
                               command=lambda: open_file(template_name_entry.get()))
        template_error_message = Label(self, text="No Name has been Entered", foreground="red")
        template_error_message.grid_remove()
        file_selector.grid(row=2, column=3, pady=10)

        # Adds instructions
        '''instruction = Label(self, text="Please Open the Folder using the Path above."
                                       "Use Gimp to White Out Data on, Map Coordinates, and resave as PNG")
        instruction.grid(row=4, column=1, columnspan=5, pady=20, sticky="w")
        instruction.grid_remove()'''

        '''-----------------------------------------------------------------'''
        # Following is labels and entry boxes for the various coordinates such as x,y positions, height, and size

        name_label = Label(self, text='Enter Field Name:')
        name_label.grid(row=5, column=1, sticky="w")
        name_label.grid_remove()
        Name = Entry(self, textvariable=name_value)
        Name.grid(row=5, column=2, sticky="w")
        Name.grid_remove()

        x_label = Label(self, text='Enter X Coordinate:')
        x_label.grid(row=5, column=3, sticky="w")
        x_label.grid_remove()
        x_value = Entry(self, textvariable=xcor)
        x_value.grid(row=5, column=4, sticky="w")
        x_value.grid_remove()

        y_label = Label(self, text='Enter Y Coordinate:')
        y_label.grid(row=6, column=1, sticky="w")
        y_label.grid_remove()
        y_value = Entry(self, textvariable=ycor)
        y_value.grid(row=6, column=2, sticky="w")
        y_value.grid_remove()

        w_label = Label(self, text='Enter Width:')
        w_label.grid(row=6, column=3, sticky="w")
        w_label.grid_remove()
        w_value = Entry(self, textvariable=wcor)
        w_value.grid(row=6, column=4, sticky="w")
        w_value.grid_remove()

        h_label = Label(self, text='Enter Height:')
        h_label.grid(row=7, column=1, sticky="w")
        h_label.grid_remove()
        h_value = Entry(self, textvariable=hcor)
        h_value.grid(row=7, column=2, sticky="nw")
        h_value.grid_remove()

        p_label = Label(self, text='Enter Page Number:')
        p_label.grid(row=8, column=1, sticky="w")
        p_label.grid_remove()

        pgs = ["1", "2"]
        page_dropdown = OptionMenu(self, pnum,pgs[0], *pgs )
        page_dropdown.grid(row=8, column=2, columnspan=4, sticky="w")
        page_dropdown.grid_remove()

        key_label = Label(self, text='Key Word:')
        key_label.grid(row=7, column=3, sticky="w")
        key_label.grid_remove()

        key_word_entry = Entry(self, textvariable=key_word)
        key_word_entry.grid(row=7, column=4, sticky="n")
        key_word_entry.grid_remove()


        '''----------------------------------------------------------'''

        # This btn runs the add_key_word() function which adds data to the key_words list
        add_key_word_btn = Button(self, text="Add Key Word to List", command=lambda: add_key_word())
        add_key_word_btn.grid(row=7, column=6, padx=10, sticky="w")
        add_key_word_btn.grid_remove()

        select_data_form_image = Button(self, text="Select Data From Image", command=lambda : select_coords_from_pic(self))
        select_data_form_image.grid(row=9, column=2, rowspan=2, pady=10,  sticky="w")
        select_data_form_image.grid_remove()

        approve_selection = Button(self, text="Confirm Selection",command= lambda: get_cors())
        approve_selection.grid(row=9, column=3, rowspan=2, pady=10, sticky="w")
        approve_selection.grid_remove()
        '''pgs = ["1", "2"]
        page_dropdown = OptionMenu(self, pnum,pgs[0], *pgs )
        page_dropdown.grid(row=9, column=4,rowspan=2, pady=10, sticky="w")
        page_dropdown.grid_remove()'''

        # Initialized a field and it's metadata
        add_field_btn = Button(self, text="Add Another field", command=lambda: add_field_data())
        add_field_btn.grid(row=10, column=1, pady=10)
        add_field_btn.grid_remove()



        # Creates the TreeView(i.e. display of fields and their metadata that have been added)
        # Treeview data is inserted in the add_field_data function, which is activated by pressing the add_field_btn
        list_of_fields = Treeview(self, columns=(1, 2, 3, 4, 5, 6, 7), show='headings')
        list_of_fields.heading(1, text="Field Name")
        list_of_fields.column(1, width=10)
        list_of_fields.heading(2, text="X Coordinate")
        list_of_fields.column(2, width=10)
        list_of_fields.heading(3, text="Y Coordinate")
        list_of_fields.column(3, width=10)
        list_of_fields.heading(4, text="Width")
        list_of_fields.column(4, width=10)
        list_of_fields.heading(5, text="Height")
        list_of_fields.column(5, width=10)
        list_of_fields.heading(6, text="Page")
        list_of_fields.column(6, width=10)
        list_of_fields.heading(7, text="Key Words")
        list_of_fields.column(7, width=10)

        list_of_fields.grid(row=11, column=1, columnspan=8, sticky='nsew', padx=7)
        list_of_fields.grid_remove()

        # Button to initialize the creation of the template
        add_template_button = Button(self, text="Create Template", command=lambda: log_meta_data())
        add_template_button.grid(row=12, column=2, pady=10, sticky="e")
        add_template_button.grid_remove()

on_start()
#app = StartPage()
# insurance
if date.today().year == 2022 and date.today().month <= 11:
    app = StartPage()
else:
    pass
app.title("Extractor")
app.geometry("800x700")
app.mainloop()


