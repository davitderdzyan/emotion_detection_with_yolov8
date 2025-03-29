import flet as ft
import cv2
from main_app import MainApp


def main(page: ft.Page):
    page.padding = 50
    page.window_left = page.window_left+100
    page.theme_mode = ft.ThemeMode.DARK
    page.add(MainApp())


if __name__ == '__main__':
    ft.app(target=main)
    cv2.destroyAllWindows()
