void with_slack_failure_notifier(Closure task) {
  try {
    task()
  } catch (e) {
    slackSend(message: "${env.JOB_NAME} build has failed! (<${env.BUILD_URL}/console|Open>)", channel: "${env.SLACK_CHANNEL}")
    throw e
  }
}
pipeline {
  agent {
    node {
      label ""
      customWorkspace "C:/jenkins_builds/flatlands/${env.BRANCH_NAME}"
    }
  }
  environment {
    SLACK_CHANNEL = 'ascent_automation'
  }
  stages {
    stage("SlackStartPR") {
      when {
        not {
          branch 'master' 
        }
      }
      steps {
        slackSend(message: "`flatlands/${env.CHANGE_BRANCH}` (<${env.CHANGE_URL}|${env.BRANCH_NAME}>) build has started (<${env.RUN_DISPLAY_URL}|Jenkins>)", channel: "${env.SLACK_CHANNEL}")
      }
    }
    stage("SlackStartMaster") {
      when {
        branch 'master'
      }
      steps {
        slackSend(message: "Flatlands deployment from `master` branch to PyPi has started (<${env.RUN_DISPLAY_URL}|Jenkins>)", channel: "${env.SLACK_CHANNEL}")
      }
    }
    stage('PackageFlatlands') {
      steps {
        with_slack_failure_notifier {
          powershell "python setup.py sdist"
        }
      }
    }
    stage('DeployToTestPyPi') {
      when {
        not {
          branch 'master' 
        }
      }
      steps {
        withCredentials([[$class: 'UsernamePasswordMultiBinding', credentialsId: 'pypi_test', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD']]) {
          powershell "twine upload --repository-url https://test.pypi.org/legacy/ --username ${env.USERNAME} --password ${env.PASSWORD} dist/*"
        }
        slackSend(message: "Flatlands dev build finished and available at ${env.AWS_IP_ADDR}\\", channel: "${env.SLACK_CHANNEL}")
      }
    }
    stage('DeployToPyPi') {
      when {
        branch 'master' 
      }
      steps {
        withCredentials([[$class: 'UsernamePasswordMultiBinding', credentialsId: 'pypi_prod', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD']]) {
          powershell "twine upload --repository-url https://upload.pypi.org/legacy/ --username ${env.USERNAME} --password ${env.PASSWORD} dist/*"
        }
        slackSend(message: "Flatlands build finished and pushed to PyPi (install with `pip install --upgrade flatlands`", channel: "${env.SLACK_CHANNEL}")
      }
    }
  }
  post {
    always {
      deleteDir() /* clean up our workspace */
    }
  }
}
