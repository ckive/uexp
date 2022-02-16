import imageio, os

# Build GIF


def build_gif(folder_path, gifname):
    bp = os.path.join('model_')
    with imageio.get_writer(gifname, mode='I') as writer:
        for filename in sorted(os.listdir(folder_path)):
            image = imageio.imread(os.path.join(folder_path, filename))
            writer.append_data(image)

        #imageio.mimsave('../animation/gif/movie.gif', images)
