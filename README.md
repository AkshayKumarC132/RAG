## Getting Started

**Version used for Project**

- **[Python 3.10.5](https://www.python.org/downloads/release/python-3105/)**
- **[Pip 22.0.4](https://pypi.org/project/pip/)**
- **[Django 5.0](https://docs.djangoproject.com/en/5.0/)**

First clone the repository from Github and switch to the new directory:

`$ git clone git@github.com/USERNAME/{{ project_name }}.git`

`$ cd {{ project_name }}`

<br>
#### Set Up Virtual Environment
Activate the virtualenv for your project :

`$ .venv\Scripts\activate`

| Note: If you are using a UNIX-based system (Linux or macOS), you may need to run source
`source .venv/bin/activate` instead.
<br>

#### Install Project Dependencies

`$ pip install -r requirements.txt`
<br>

#### Database Migration

Run the following command to apply migrations :-

`$ py manage.py migrate`
<br>

#### Run the Development Server

You can now run the development server:-

`$ python manage.py runserver `

Your development server will be accessible at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
