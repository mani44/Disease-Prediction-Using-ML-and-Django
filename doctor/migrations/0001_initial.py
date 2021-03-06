# Generated by Django 3.0.6 on 2020-05-30 13:27

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='copd',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('patientemail', models.CharField(max_length=100)),
                ('docname', models.CharField(max_length=100)),
                ('reportof', models.CharField(max_length=100)),
                ('reportnm', models.CharField(max_length=100)),
                ('lipcolor', models.CharField(max_length=100)),
                ('FEV', models.CharField(max_length=100)),
                ('Smkintensity', models.CharField(max_length=100)),
                ('temp', models.CharField(max_length=100)),
                ('riskvalue', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='diabetesreport',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('patientemail', models.CharField(max_length=100)),
                ('docname', models.CharField(max_length=100)),
                ('reportof', models.CharField(max_length=100)),
                ('reportnm', models.CharField(max_length=100)),
                ('glucose', models.CharField(max_length=100)),
                ('bloodpressure', models.CharField(max_length=100)),
                ('insulin', models.CharField(max_length=100)),
                ('bmi', models.CharField(max_length=100)),
                ('diapedgree', models.CharField(max_length=100)),
                ('riskvalue', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='DoctorReg',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pname', models.CharField(max_length=100)),
                ('pemail', models.CharField(max_length=100)),
                ('pphone', models.CharField(max_length=100)),
                ('paddress', models.TextField()),
                ('password', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Heartreport',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('patientemail', models.CharField(max_length=100)),
                ('docname', models.CharField(max_length=100)),
                ('reportof', models.CharField(max_length=100)),
                ('reportnm', models.CharField(max_length=100)),
                ('cp', models.CharField(max_length=100)),
                ('trestbps', models.CharField(max_length=100)),
                ('chol', models.CharField(max_length=100)),
                ('fbs', models.CharField(max_length=100)),
                ('exang', models.CharField(max_length=100)),
                ('ca', models.CharField(max_length=100)),
                ('riskvalue', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='PatientReg',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pname', models.CharField(max_length=100)),
                ('pemail', models.CharField(max_length=100)),
                ('pphone', models.CharField(max_length=100)),
                ('paddress', models.TextField()),
                ('password', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='ReportPredict',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Occupational_hazard', models.IntegerField()),
                ('Genetic_Risk', models.IntegerField()),
                ('chronic_lung_cancer', models.IntegerField()),
                ('smocking', models.IntegerField()),
                ('passive_smoker', models.IntegerField()),
                ('chest_pain', models.IntegerField()),
                ('coughing_of_blood', models.IntegerField()),
                ('fatigue', models.IntegerField()),
                ('weight_loss', models.IntegerField()),
                ('dry_cough', models.IntegerField()),
                ('clubbing_of_finger_nail', models.IntegerField()),
            ],
        ),
    ]
