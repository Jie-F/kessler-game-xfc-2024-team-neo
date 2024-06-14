import os
from tkinter import *
from PIL import Image, ImageTk

from .graphics_base import KesslerGraphics

SCALE = 2

class GraphicsTK(KesslerGraphics):
    def __init__(self, UI_settings):
        UI_settings = {} if UI_settings is None else UI_settings
        self.show_ships = UI_settings.get('ships', True)
        self.show_lives = UI_settings.get('lives_remaining', True)
        self.show_accuracy = UI_settings.get('accuracy', True)
        self.show_asteroids_hit = UI_settings.get('asteroids_hit', True)
        self.show_shots_fired = UI_settings.get('shots_fired', False)
        self.show_bullets_remaining = UI_settings.get('bullets_remaining', True)
        self.show_mines_remaining = UI_settings.get('mines_remaining', True)
        self.show_controller_name = UI_settings.get('controller_name', True)

    def start(self, scenario):
        self.game_width = round(scenario.map_size[0] * SCALE)
        self.game_height = round(scenario.map_size[1] * SCALE)
        self.max_time = scenario.time_limit
        self.score_width = round(485 * SCALE)
        self.window_width = round(self.game_width + self.score_width)
        ship_radius = round((scenario.ships()[0].radius * 2 - 5) * SCALE)

        self.window = Tk()
        self.window.title('Kessler')
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        center_x = int(screen_width / 2 - self.window_width / 2)
        center_y = int(screen_height / 2 - self.game_height / 2)
        self.window.geometry(f'{self.window_width}x{self.game_height}+{center_x}+{center_y}')

        self.game_canvas = Canvas(self.window, width=self.window_width, height=self.game_height, bg="black")
        self.game_canvas.pack()
        self.window.update()

        script_dir = os.path.dirname(__file__)
        self.image_paths = ["images/playerShip1_green.png", "images/playerShip1_orange.png", "images/playerShip2_orange.png", "images/playerShip3_orange.png"]
        self.num_images = len(self.image_paths)
        self.ship_images = [(Image.open(os.path.join(script_dir, image))).resize((ship_radius, ship_radius)) for image in self.image_paths]
        self.ship_sprites = [ImageTk.PhotoImage(img) for img in self.ship_images]
        self.ship_icons = [ImageTk.PhotoImage((Image.open(os.path.join(script_dir, image))).resize((ship_radius, ship_radius))) for image in self.image_paths]

        self.detoantion_time = 0.3
        self.detonation_timers = []

    def update(self, score, ships, asteroids, bullets, mines):
        self.game_canvas.delete("all")

        self.plot_shields(ships)
        self.plot_ships(ships)
        self.plot_bullets(bullets)
        self.plot_asteroids(asteroids)
        self.plot_mines(mines)

        self.update_score(score, ships)

        self.window.update()

    def close(self):
        self.window.destroy()

    def update_score(self, score, ships):
        x_offset = round(5 * SCALE)
        y_offset = round(5 * SCALE)

        self.game_canvas.create_rectangle(self.game_width, 0, self.window_width, self.game_height, outline="white", fill="black",)
        self.game_canvas.create_line(self.window_width - self.score_width / 2, 0, self.window_width - self.score_width / 2, self.game_height, fill="white")

        font_size = round(10 * SCALE)  # Increased font size for better readability
        time_text = "Time: " + f'{score.sim_time:.2f}' + " / " + str(self.max_time) + " sec"
        self.game_canvas.create_text(round(10 * SCALE), round(10 * SCALE), text=time_text, fill="white", font=("Courier New", font_size), anchor=NW)

        team_num = 0
        output_location_y = 0
        max_lines = 0

        for team in score.teams:
            title = team.team_name + "\n"
            ships_text = "_________\n"

            if self.show_ships:
                for ship in ships:
                    if ship.team == team.team_id:
                        ships_text += "Ship " + str(ship.id)
                        if self.show_controller_name:
                            ships_text += ": " + str(ship.controller.name)
                        ships_text += '\n'

            team_info = self.format_ui(team)
            score_board = title + ships_text + team_info

            if (team_num % 2) == 0:
                output_location_x = self.game_width + x_offset
                output_location_y = output_location_y + (17 * max_lines) + y_offset
                self.game_canvas.create_line(self.game_width, output_location_y - 10 * SCALE, self.window_width, output_location_y - 10 * SCALE, fill="white")
                max_lines = score_board.count("\n")
            else:
                output_location_x = self.window_width + x_offset - self.score_width / 2
                if score_board.count("\n") > max_lines:
                    max_lines = score_board.count("\n")

            self.game_canvas.create_text(output_location_x, output_location_y, text=score_board, fill="white", font=("Courier New", font_size), anchor=NW)
            self.game_canvas.create_image(output_location_x + 120 * SCALE, output_location_y + 15 * SCALE, image=self.ship_icons[(team.team_id-1) % self.num_images])
            team_num += 1

    def format_ui(self, team):
        team_info = "_________\n"
        if self.show_lives:
            team_info += "Lives: " + str(team.lives_remaining) + "\n"
        if self.show_accuracy:
            team_info += "Accuracy: " + str(round(team.accuracy * 100, 1)) + "%\n"
        if self.show_asteroids_hit:
            team_info += "Asteroids Hit: " + str(team.asteroids_hit) + "\n"
        if self.show_shots_fired:
            team_info += "Shots Fired: " + str(team.shots_fired) + "\n"
        if self.show_bullets_remaining:
            team_info += "Bullets Left: " + str(team.bullets_remaining) + "\n"
        if self.show_mines_remaining:
            team_info += "Mines Left: " + str(team.mines_remaining) + "\n"
        return team_info

    def plot_ships(self, ships):
        for idx, ship in enumerate(ships):
            if ship.alive:
                self.ship_sprites[idx] = ImageTk.PhotoImage(self.ship_images[idx].rotate(180 - (-ship.heading - 90)))
                self.game_canvas.create_image(ship.position[0]*SCALE, self.game_height - ship.position[1]*SCALE, image=self.ship_sprites[idx])
                self.game_canvas.create_text(ship.position[0]*SCALE + ship.radius * SCALE, self.game_height - (ship.position[1]*SCALE + ship.radius * SCALE), text=str(ship.id), fill="white", font=("Courier New", 10))

    def plot_shields(self, ships):
        for ship in ships:
            if ship.alive:
                respawn_scaler = max(min(ship.respawn_time_left, 1), 0)
                r = int(120 + (respawn_scaler * (255 - 120)))
                g = int(200 + (respawn_scaler * (0 - 200)))
                b = int(255 + (respawn_scaler * (0 - 255)))
                color = "#%02x%02x%02x" % (r, g, b)
                self.game_canvas.create_oval(ship.position[0]*SCALE - ship.radius * SCALE, self.game_height - (ship.position[1]*SCALE + ship.radius * SCALE), ship.position[0]*SCALE + ship.radius * SCALE, self.game_height - (ship.position[1]*SCALE - ship.radius * SCALE), fill="black", outline=color)

    def plot_bullets(self, bullets):
        for bullet in bullets:
            self.game_canvas.create_line(bullet.position[0]*SCALE, self.game_height - bullet.position[1]*SCALE, bullet.tail[0]*SCALE, self.game_height - bullet.tail[1]*SCALE, fill="#EE2737", width=6)  # Increased line width

    def plot_asteroids(self, asteroids):
        for asteroid in asteroids:
            self.game_canvas.create_oval(asteroid.position[0]*SCALE - asteroid.radius * SCALE, self.game_height - (asteroid.position[1]*SCALE + asteroid.radius * SCALE), asteroid.position[0]*SCALE + asteroid.radius * SCALE, self.game_height - (asteroid.position[1]*SCALE - asteroid.radius * SCALE), fill="grey")

    def plot_mines(self, mines):
        for mine in mines:
            self.game_canvas.create_oval(mine.position[0]*SCALE - mine.radius * SCALE, self.game_height - (mine.position[1]*SCALE + mine.radius * SCALE), mine.position[0]*SCALE + mine.radius * SCALE, self.game_height - (mine.position[1]*SCALE - mine.radius * SCALE), fill="yellow")
            light_fill = "red" if mine.countdown_timer - int(mine.countdown_timer) > 0.5 else "orange"
            self.game_canvas.create_oval(mine.position[0]*SCALE - mine.radius * 0.6 * SCALE, self.game_height - (mine.position[1]*SCALE + mine.radius * 0.6 * SCALE), mine.position[0]*SCALE + mine.radius * 0.6 * SCALE, self.game_height - (mine.position[1]*SCALE - mine.radius * 0.6 * SCALE), fill=light_fill)
            if mine.countdown_timer < mine.detonation_time:
                explosion_radius = mine.blast_radius * (1 - mine.countdown_timer / mine.detonation_time)**2 * SCALE
                self.game_canvas.create_oval(mine.position[0]*SCALE - explosion_radius, self.game_height - (mine.position[1]*SCALE + explosion_radius), mine.position[0]*SCALE + explosion_radius, self.game_height - (mine.position[1]*SCALE - explosion_radius), fill="", outline="white", width=20)
