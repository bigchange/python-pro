#!/usr/bin/env python
# -*- coding:utf-8 -*-

# File: match_model.py
# Project: /Users/tech/code/idmg
# Created: Thu Aug 03 2017
# Author: Koth Chen
# Copyright (c) 2017 Koth
#
# <<licensetext>>

import tensorflow as tf
import math

# tfrecord数据装换
def parse_tfrecord_function(example_proto):
    totalTags = 1930
    majorVal = 0.9
    defaultVal = 0.1 / (totalTags - 1)
    features = {
        "target": tf.FixedLenFeature([], tf.int64,
                                     default_value=0),
        "target_orgId":tf.FixedLenFeature([], tf.int64,
                                     default_value=0),
        "gender": tf.FixedLenFeature([], tf.int64,
                                     default_value=0),
        "age": tf.FixedLenFeature([], tf.int64, default_value=0),
        "location": tf.FixedLenFeature([], tf.int64,
                                       default_value=0),
        "education_schools": tf.FixedLenSequenceFeature([],
                                                        tf.int64,
                                                        allow_missing=True,
                                                        default_value=0),
        "education_degrees": tf.FixedLenSequenceFeature([],
                                                        tf.int64,
                                                        allow_missing=True,
                                                        default_value=0),
        "education_starts": tf.FixedLenSequenceFeature([],
                                                       tf.float32,
                                                       allow_missing=True,
                                                       default_value=0),
        "education_majors": tf.FixedLenSequenceFeature([],
                                                       tf.int64,
                                                       allow_missing=True,
                                                       default_value=0),
        "work_expr_descs": tf.FixedLenSequenceFeature([],
                                                      tf.int64,
                                                      allow_missing=True,
                                                      default_value=0),
        "work_expr_orgs": tf.FixedLenSequenceFeature([],
                                                     tf.int64,
                                                     allow_missing=True,
                                                     default_value=0),
        "work_expr_orgIds": tf.FixedLenSequenceFeature([],
                                                     tf.int64,
                                                     allow_missing=True,
                                                     default_value=0),
        "work_expr_starts": tf.FixedLenSequenceFeature([],
                                                       tf.float32,
                                                       allow_missing=True,
                                                       default_value=0),
        "work_expr_durations": tf.FixedLenSequenceFeature([],
                                                          tf.float32,
                                                          allow_missing=True,
                                                          default_value=0),
        "work_expr_jobs": tf.FixedLenSequenceFeature([],
                                                     tf.int64,
                                                     allow_missing=True,
                                                     default_value=0),
        "proj_expr_descs": tf.FixedLenSequenceFeature([],
                                                      tf.int64,
                                                      allow_missing=True,
                                                      default_value=0),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    target = parsed_features["target"]
    target_orgId = parsed_features["target_orgId"]
    targets = tf.sparse_to_dense(target, [totalTags], majorVal, defaultVal)
    gender = parsed_features["gender"]
    age = parsed_features["age"]
    location = parsed_features["location"]
    # education part
    education_schools = parsed_features["education_schools"]
    education_schools.set_shape([3])
    education_degrees = parsed_features["education_degrees"]
    education_degrees.set_shape([3])
    education_starts = parsed_features["education_starts"]
    education_starts.set_shape([3])
    education_majors = parsed_features["education_majors"]
    education_majors.set_shape([3])
    # working experience part
    work_expr_orgs = parsed_features["work_expr_orgs"]
    work_expr_orgs.set_shape([3])
    work_expr_starts = parsed_features["work_expr_starts"]
    work_expr_starts.set_shape([3])
    work_expr_durations = parsed_features["work_expr_durations"]
    work_expr_durations.set_shape([3])
    work_expr_orgIds = parsed_features["work_expr_orgIds"]
    work_expr_orgIds.set_shape([3])
    work_expr_jobs = parsed_features["work_expr_jobs"]
    work_expr_jobs.set_shape([3])
    work_expr_descs = parsed_features["work_expr_descs"]
    work_expr_descs.set_shape([360])
    work_expr_descs = tf.reshape(work_expr_descs, [3, 120])

    proj_expr_descs = parsed_features["proj_expr_descs"]
    proj_expr_descs.set_shape([360])
    proj_expr_descs = tf.reshape(proj_expr_descs, [3, 120])

    return target, targets, gender, age, location, education_schools, education_degrees, education_starts, education_majors, work_expr_orgs, work_expr_starts, work_expr_durations, work_expr_jobs, work_expr_orgIds, work_expr_descs, proj_expr_descs

# embedding ids
def get_embedding(maxId, dimSize, name):
    return tf.get_variable(shape=[maxId, dimSize],
                           initializer=tf.random_uniform_initializer(0, 0.05),
                           name=name)


def calc_xvar_init(n, m):
    return math.sqrt(6.0 / (n + m))


class MatchModel:
    def __init__(self,
                 words_,
                 lstmEmSize_=150,
                 max_age_id=11,
                 age_dim=10,
                 max_degree_id=7,
                 degree_dim=4,
                 max_loc_id=3210,
                 loc_dim=20,
                 max_school_id=2751,
                 school_dim=20,
                 max_major_id=1458,
                 major_dim=20,
                 max_org_id=4,
                 org_dim=10,
                 max_job_id=1930,
                 job_dim=30,
                 max_orgId=7516,
                 org_id_dim=30,
                 max_seq_len=120,
                 attention_query_size=150,
                 last_hidden_size_=2048,
                 num_class=1930,
                 jobEmbeddings=None):
        self.words = tf.Variable(words_, name="words")
        self.lstmEmSize = lstmEmSize_
        self.embAge = get_embedding(max_age_id, age_dim, "age_embedding")
        self.embOrgID = get_embedding(max_orgId, org_id_dim, "org_id_embedding")
        self.embDegree = get_embedding(max_degree_id, degree_dim,
                                       "degree_embedding")
        self.embLocation = get_embedding(max_loc_id, loc_dim,
                                         "location_embedding")
        self.embSchool = get_embedding(max_school_id, school_dim,
                                       "school_embedding")
        self.embMajor = get_embedding(max_major_id, major_dim,
                                      "major_embedding")
        self.embOrg = get_embedding(max_org_id, org_dim, "org_embedding")
        if jobEmbeddings is None:
            self.embJob = get_embedding(max_job_id, job_dim, "job_embedding")
        else:
            self.embJob = tf.Variable(jobEmbeddings, name="job_embedding")
        self.maxSeqLen = max_seq_len
        xv = calc_xvar_init(self.lstmEmSize * 2, attention_query_size)
        self.word_contex_weight = tf.get_variable(
            "word_contex_weight",
            shape=[self.lstmEmSize * 2, attention_query_size],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.random_uniform_initializer(minval=0 - xv,
                                                      maxval=xv))

        self.word_contex_weight_top = tf.get_variable(
            "word_contex_weight_top",
            shape=[self.lstmEmSize * 2, attention_query_size],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.random_uniform_initializer(minval=0 - xv,
                                                      maxval=xv))
        self.word_contex_weight_top_proj = tf.get_variable(
            "word_contex_weight_top_proj",
            shape=[self.lstmEmSize * 2, attention_query_size],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.random_uniform_initializer(minval=0 - xv,
                                                      maxval=xv))

        self.word_contex_weight_proj = tf.get_variable(
            "word_contex_weight_proj",
            shape=[self.lstmEmSize * 2, attention_query_size],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.random_uniform_initializer(minval=0 - xv,
                                                      maxval=xv))

        self.word_contex_bias = tf.get_variable("word_contex_bias",
                                                shape=[attention_query_size])

        self.word_contex_bias_top = tf.get_variable(
            "word_contex_bias_top",
            shape=[attention_query_size])

        self.word_contex_bias_top_proj = tf.get_variable(
            "word_contex_bias_top_proj",
            shape=[attention_query_size])

        self.word_contex_bias_proj = tf.get_variable("word_contex_bias_proj",
                                                     shape=[attention_query_size])

        self.word_contex = tf.get_variable(
            "word_contex",
            shape=[attention_query_size],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.random_uniform_initializer(minval=0 - 0.5,
                                                      maxval=0.5))
        self.word_contex_top = tf.get_variable(
            "word_contex_top",
            shape=[attention_query_size],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.random_uniform_initializer(minval=0 - 0.5,
                                                      maxval=0.5))
        self.word_contex_top_proj = tf.get_variable(
            "word_contex_top_proj",
            shape=[attention_query_size],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.random_uniform_initializer(minval=0 - 0.5,
                                                      maxval=0.5))

        self.word_contex_proj = tf.get_variable(
            "word_contex_proj",
            shape=[attention_query_size],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.random_uniform_initializer(minval=0 - 0.5,
                                                      maxval=0.5))

        self.last_hidden_size = last_hidden_size_
        
        self.label_target_holder = tf.placeholder(tf.int32, shape=[None])
        self.targets_holder = tf.placeholder(tf.float32,
                                             shape=[None, num_class])
        self.gender_holder = tf.placeholder(
            tf.int32, shape=[None], name="gender_holder")
        self.age_holder = tf.placeholder(
            tf.int32, shape=[None], name="age_holder")
        self.location_holder = tf.placeholder(
            tf.int32, shape=[None], name="location_holder")
        self.education_schools_holder = tf.placeholder(tf.int32,
                                                       shape=[None, 3], name="eshools_holder")
        self.education_degrees_holder = tf.placeholder(tf.int32,
                                                       shape=[None, 3], name="edegrees_holder")
        self.education_starts_holder = tf.placeholder(tf.float32,
                                                      shape=[None, 3], name="estarts_holder")
        self.education_majors_holder = tf.placeholder(tf.int32,
                                                      shape=[None, 3], name="emajors_holder")
        self.work_expr_orgs_holder = tf.placeholder(
            tf.int32, shape=[None, 3], name="worgs_holder")
        self.work_expr_starts_holder = tf.placeholder(tf.float32,
                                                      shape=[None, 3], name="wstarts_holder")
        self.work_expr_durations_holder = tf.placeholder(tf.float32,
                                                         shape=[None, 3], name="wdurations_holder")
        self.work_expr_jobs_holder = tf.placeholder(
            tf.int32, shape=[None, 3], name="wjobs_holder")

        self.work_expr_orgIds_holder = tf.placeholder(
            tf.int32, shape=[None, 3], name="wOrgIDs_holder")
        self.work_expr_descs_holder = tf.placeholder(
            tf.int32, shape=[None, 3, max_seq_len], name="wdescs_holder")
        self.proj_expr_descs_holder = tf.placeholder(
            tf.int32, shape=[None, 3, max_seq_len], name="pdescs_holder")
        self.learning_rate_h = tf.placeholder(tf.float32, shape=[])
        self.part_weights = tf.get_variable(
            "proj_part_weights",
            shape=[],
            initializer=tf.constant_initializer(0.5))
        self.num_class =num_class
        self.basic_feature_hidden=128
        self.final_weights_bias = tf.get_variable("final_weights_bias",
                                                        shape=[self.num_class])

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

        # 组合基本特征
    def modelBasicFeatures(self,
                           gender,
                           age,
                           location,
                           education_schools,
                           education_degrees,
                           education_starts,
                           education_majors,
                           reuse=False):
        with tf.variable_scope("basic_feature", reuse=reuse):
            genderF = tf.cast(gender, tf.float32) - 1.0
            genderF = tf.expand_dims(genderF, -1)
            ageF = tf.nn.embedding_lookup(self.embAge, age)
            locF = tf.nn.embedding_lookup(self.embLocation, location)
            eduSchoolFs = tf.nn.embedding_lookup(self.embSchool,
                                                 education_schools)
            eduDegreeFs = tf.nn.embedding_lookup(self.embDegree,
                                                 education_degrees)
            eduMajorFs = tf.nn.embedding_lookup(self.embMajor,
                                                education_majors)
            finalOut = tf.concat([genderF, ageF, locF], axis=1)
            edSchoolsDims = eduSchoolFs.get_shape().as_list()
            eduSchoolFs = tf.reshape(eduSchoolFs,
                                     [-1, edSchoolsDims[1] * edSchoolsDims[2]])

            eduDegreeDims = eduDegreeFs.get_shape().as_list()
            eduDegreeFs = tf.reshape(eduDegreeFs,
                                     [-1, eduDegreeDims[1] * eduDegreeDims[2]])

            eduMajorDims = eduMajorFs.get_shape().as_list()
            eduMajorFs = tf.reshape(eduMajorFs,
                                    [-1, eduMajorDims[1] * eduMajorDims[2]])
            #Following code is for crossnetwork
            X0 = tf.concat([finalOut, eduSchoolFs, eduDegreeFs,
                                  eduMajorFs],
                                 axis=1)
            basicLastDim = X0.get_shape().as_list()[1]
            
            self.cross_weight=tf.get_variable(
                "basic_cross_weight",
                shape=[basicLastDim,1],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.contrib.layers.xavier_initializer())
            self.cross_bias =tf.get_variable(
                "basic_cross_bias",
                shape=[basicLastDim],
                initializer=tf.random_uniform_initializer(-0.01,0.01))
            Xt= tf.expand_dims(X0,1)
            X1=tf.matmul(tf.reshape(tf.matmul(tf.expand_dims(X0,-1),Xt),[-1,basicLastDim]),self.cross_weight)
            X1=tf.reshape(tf.squeeze(X1),[-1,basicLastDim])+X0+self.cross_bias

            basicFeatureWeight = tf.get_variable(
                "basic_feature_weight",
                shape=[basicLastDim, self.basic_feature_hidden],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.contrib.layers.xavier_initializer())
            basicFeatureBias = tf.get_variable("basic_feature_bias",
                                               shape=[self.basic_feature_hidden],
                                               initializer=tf.random_uniform_initializer(-0.01,0.01))
            Xh=tf.nn.xw_plus_b(X0,basicFeatureWeight,basicFeatureBias)
            Xh=tf.nn.elu(Xh)
            X = tf.concat([X1, Xh],
                                 axis=1)
            return X

    def do_attention(self, X, dropout=False, name=None, forProj=False):
        Xr = tf.reshape(X, [-1, self.lstmEmSize * 2])
        C1 = tf.nn.xw_plus_b(Xr, self.word_contex_weight_proj if forProj else self.word_contex_weight,
                             self.word_contex_bias_proj if forProj else self.word_contex_bias)
        C1 = tf.nn.elu(C1)
        if dropout:
            C1 = tf.nn.dropout(C1, 0.7)
        a1 = tf.matmul(C1, tf.expand_dims(
            self.word_contex_proj if forProj else self.word_contex, -1))
        a1 = tf.reshape(a1, [-1, self.maxSeqLen])
        retA = tf.nn.softmax(a1, name=None if name is None else name)
        a1 = tf.expand_dims(retA, dim=1)
        L1 = tf.squeeze(tf.matmul(a1, X), squeeze_dims=1)
        return L1

    def do_attention_top(self, X, dropout=False, name=None, forProj=False):
        Xr = tf.reshape(X, [-1, self.lstmEmSize * 2])
        C1 = tf.nn.xw_plus_b(Xr, self.word_contex_weight_top_proj if forProj else self.word_contex_weight_top,
                             self.word_contex_bias_top_proj if forProj else self.word_contex_bias_top)
        C1 = tf.nn.elu(C1)
        if dropout:
            C1 = tf.nn.dropout(C1, 0.7)
        a1 = tf.matmul(C1, tf.expand_dims(
            self.word_contex_top_proj if forProj else self.word_contex_top, -1))
        a1 = tf.reshape(a1, [-1, 3])
        retA = tf.nn.softmax(a1, name=None if name is None else name)
        a1 = tf.expand_dims(retA, dim=1)
        L1 = tf.squeeze(tf.matmul(a1, X), squeeze_dims=1)
        return L1

    def do_bilstm(self, X, reuse=False, name=None, forProj=False):
        seqEmds = tf.nn.embedding_lookup(self.words, X)
        length = self.length(X)
        length_64 = tf.cast(length, tf.int64)
        dropout = False
        if reuse is None or not reuse:
            seqEmds = tf.nn.dropout(seqEmds, keep_prob=0.7)
            dropout = True
        with tf.variable_scope("rnn_fwbw", reuse=reuse):
            rnn_cell_f = tf.contrib.rnn.LSTMCell(self.lstmEmSize, reuse=reuse)
            rnn_cell_b = tf.contrib.rnn.LSTMCell(self.lstmEmSize, reuse=reuse)
            forward_output, _ = tf.nn.dynamic_rnn(rnn_cell_f,
                                                  seqEmds,
                                                  dtype=tf.float32,
                                                  time_major=False,
                                                  sequence_length=length,
                                                  scope="RNN_forword")
            backward_output_, _ = tf.nn.dynamic_rnn(rnn_cell_b,
                                                    inputs=tf.reverse_sequence(
                                                        seqEmds,
                                                        length_64,
                                                        batch_dim=0,
                                                        seq_dim=1),
                                                    dtype=tf.float32,
                                                    time_major=False,
                                                    sequence_length=length,
                                                    scope="RNN_backword")
            backward_output = tf.reverse_sequence(backward_output_,
                                                  length_64,
                                                  batch_dim=0,
                                                  seq_dim=1)
        R = tf.concat([forward_output, backward_output], 2)
        return self.do_attention(R, dropout, name=name, forProj=forProj)
    # 组合工作经历特征

    def modelWorkingExperience(self,
                               work_expr_orgs,
                               work_expr_starts,
                               work_expr_durations,
                               work_expr_jobs,
                               work_expr_orgIds,
                               work_expr_descs,
                               reuse=False):
        with tf.variable_scope("working_expr", reuse=reuse):
            workOrgs = tf.nn.embedding_lookup(self.embOrg, work_expr_orgs)
            workJobs = tf.nn.embedding_lookup(self.embJob, work_expr_jobs)
            workOrgID = tf.nn.embedding_lookup(self.embOrgID, work_expr_orgIds)
            weds = tf.reshape(work_expr_descs, [-1, self.maxSeqLen])
            weds = self.do_bilstm(weds, reuse, name="work_desc_alignment" if (
                reuse is not None and reuse) else None)
            weds = tf.reshape(weds, [-1, 3, self.lstmEmSize * 2])
            dropout = False
            if reuse is None or not reuse:
                dropout = True
            weds = self.do_attention_top(weds, dropout)

            workOrgsDims = workOrgs.get_shape().as_list()
            workOrgsFs = tf.reshape(workOrgs,
                                    [-1, workOrgsDims[1] * workOrgsDims[2]])

            workOrgIDDims = workOrgID.get_shape().as_list()
            workOrgIdFs =  tf.reshape(workOrgID,
                                    [-1, workOrgIDDims[1] * workOrgIDDims[2]])                 
            workJobsDims = workJobs.get_shape().as_list()
            workJobsFs = tf.reshape(workJobs,
                                    [-1, workJobsDims[1] * workJobsDims[2]])

            finalOut = tf.concat([workOrgsFs, workJobsFs, workOrgIdFs, weds],
                                 axis=1)
            # if dropout:
            #     finalOut = tf.nn.dropout(finalOut, 0.5)
            return finalOut

    # 组合项目经历特征
    def modelProjExperience(self, proj_expr_descs, reuse=False):
        # reuse set to true to reuse the variable as working experiences
        with tf.variable_scope("proj_expr", reuse=reuse):
            peds = tf.reshape(proj_expr_descs, [-1, self.maxSeqLen])
            peds = self.do_bilstm(peds, reuse, name="proj_desc_alignment" if (
                reuse is not None and reuse) else None, forProj=True)
            peds = tf.reshape(peds, [-1, 3, self.lstmEmSize * 2])
            dropout = False
            if reuse is None or not reuse:
                dropout = True
            peds = self.do_attention_top(peds, dropout, forProj=True)
            return peds

    def inference(self,
                  gender,
                  age,
                  location,
                  education_schools,
                  education_degrees,
                  education_starts,
                  education_majors,
                  work_expr_orgs,
                  work_expr_starts,
                  work_expr_durations,
                  work_expr_jobs,
                  work_expr_orgIds,
                  work_expr_descs,
                  proj_expr_descs,
                  reuse=False):
        basicFeatures = self.modelBasicFeatures(gender,
                                                age,
                                                location,
                                                education_schools,
                                                education_degrees,
                                                education_starts,
                                                education_majors,
                                                reuse=reuse)
        workFeatures = self.modelWorkingExperience(work_expr_orgs,
                                                   work_expr_starts,
                                                   work_expr_durations,
                                                   work_expr_jobs,
                                                   work_expr_orgIds,
                                                   work_expr_descs,
                                                   reuse=reuse)

        projFeatures = self.modelProjExperience(proj_expr_descs, reuse=reuse)
        workLastDim = workFeatures.get_shape().as_list()[1]
        projLastDim = projFeatures.get_shape().as_list()[1]
        basicLastDim =basicFeatures.get_shape().as_list()[1]
        if reuse is None or not reuse:
            print("work last dim:%d" %
                  (workLastDim))
        with tf.variable_scope("final_layers", reuse=reuse):
            
            xv = calc_xvar_init(workLastDim, self.last_hidden_size)
            workFeatureWeight = tf.get_variable(
                "work_feature_weight",
                shape=[workLastDim, self.last_hidden_size],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.random_uniform_initializer(minval=0 - xv,
                                                          maxval=xv))
            workFeatureBias = tf.get_variable("work_feature_bias",
                                              shape=[self.last_hidden_size])

            xv = calc_xvar_init(projLastDim, self.last_hidden_size)
            projFeatureWeight = tf.get_variable(
                "proj_feature_weight",
                shape=[projLastDim, self.last_hidden_size],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.random_uniform_initializer(minval=0 - xv,
                                                          maxval=xv))
            projFeatureBias = tf.get_variable("proj_feature_bias",
                                              shape=[self.last_hidden_size])
            sumWeight = 1.0 + self.part_weights
            # finalFeaturesBasic = tf.nn.relu(tf.nn.xw_plus_b(
            #     basicFeatures, basicFeatureWeight, basicFeatureBias))

            finalFeaturesWork = tf.nn.relu(tf.nn.xw_plus_b(
                workFeatures, workFeatureWeight, workFeatureBias))
            finalFeaturesProj = tf.nn.relu(tf.nn.xw_plus_b(
                projFeatures, projFeatureWeight, projFeatureBias))

            finalFeatureText = tf.add(
                finalFeaturesWork, finalFeaturesProj * (self.part_weights / sumWeight))
            finalFeatureDims=basicLastDim+self.last_hidden_size
            xv = calc_xvar_init(finalFeatureDims, self.num_class)
            self.final_weights = tf.get_variable(
                "final_weights",
                shape=[finalFeatureDims, self.num_class],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.random_uniform_initializer(minval=0 - xv,
                                                        maxval=xv))
            finalFeatures = tf.concat(
                [basicFeatures, finalFeatureText], axis=1, name="finalFeatures")
            finalFeatures =tf.nn.elu(finalFeatures)
            if reuse is None or not reuse:
                finalFeatures = tf.nn.dropout(finalFeatures, 0.7)
            return tf.nn.xw_plus_b(finalFeatures,
                                   self.final_weights,
                                   self.final_weights_bias,
                                   name="finalOut"
                                   if (reuse is not None and reuse) else None)

    def train(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_h)
        return optimizer.minimize(loss)

    def test_loss(self):
        return self.loss(
            [self.label_target_holder, self.targets_holder, self.gender_holder,
             self.age_holder, self.location_holder,
             self.education_schools_holder, self.education_degrees_holder,
             self.education_starts_holder, self.education_majors_holder,
             self.work_expr_orgs_holder, self.work_expr_starts_holder,
             self.work_expr_durations_holder, self.work_expr_jobs_holder,self.work_expr_orgIds_holder,
             self.work_expr_descs_holder, self.proj_expr_descs_holder],
            reuse=True)

    def loss(self, inputs, reuse=False):
        target, targets, gender, age, location, education_schools, education_degrees, education_starts, education_majors, work_expr_orgs, work_expr_starts, work_expr_durations, work_expr_jobs, work_expr_orgIds, work_expr_descs, proj_expr_descs = inputs[
            0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[
                6], inputs[7], inputs[8], inputs[9], inputs[10], inputs[
                    11], inputs[12], inputs[13],inputs[14], inputs[15]
        preds = self.inference(gender,
                               age,
                               location,
                               education_schools,
                               education_degrees,
                               education_starts,
                               education_majors,
                               work_expr_orgs,
                               work_expr_starts,
                               work_expr_durations,
                               work_expr_jobs,
                               work_expr_orgIds,
                               work_expr_descs,
                               proj_expr_descs,
                               reuse=reuse)
        top3 = tf.nn.in_top_k(preds, target, 3)
        top1 = tf.nn.in_top_k(preds, target, 1)
        top10 = tf.nn.in_top_k(preds, target, 10)
        t3 = tf.reduce_sum(tf.cast(top3, tf.int32))
        t1 = tf.reduce_sum(tf.cast(top1, tf.int32))
        t10 = tf.reduce_sum(tf.cast(top10, tf.int32))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=targets, logits=preds))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        cost += tf.reduce_sum(reg_losses)
        return cost, t1, t3, t10
