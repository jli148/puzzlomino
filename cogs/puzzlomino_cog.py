import io

import cv2
from discord.ext import commands
from discord import File
import matplotlib.pyplot as plt
import numpy as np
import requests

from puzzlomino import puzzlomino


class PuzzlominoCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def puzzle(self, ctx: commands.Context):
        if not ctx.message.attachments:
            await ctx.send("Please attach an image of the puzzle")

        await ctx.send("Calculating...")

        original = url_to_img(ctx.message.attachments[0].url)

        processed = puzzlomino.preprocess(original)
        puzzle_contour = puzzlomino.get_puzzle_contour(processed)

        completed = puzzle_contour.area()
        contour_img = puzzle_contour.overlay_on(original)
        attachment = img_to_file(contour_img, filename="contour.png")

        await ctx.send(
            content=f"Puzzle is approximately {completed:0.1%} complete",
            file=attachment,
        )


def url_to_img(url: str) -> cv2.typing.MatLike:
    resp = requests.get(url, stream=True).raw
    bytes = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(bytes, cv2.IMREAD_COLOR)

    return img


def img_to_file(img: cv2.typing.MatLike, filename: str) -> File:
    buf = io.BytesIO()
    plt.imsave(buf, img)
    plt.close()
    buf.seek(0)

    contour_img = File(buf, filename)
    buf.close()

    return contour_img


async def setup(bot):
    await bot.add_cog(PuzzlominoCog(bot))
