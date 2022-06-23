'use strict';

class Lesson extends React.Component {
  constructor(props) {
    console.log(props)
    super(props);
    this.state = {
      lessonId: props.lessonId,
      name: props.name,
      description: props.description
    }
  }

  loadLession() {
    let lessons = new Map()
    lessons.set("lesson1", <Lesson1 />)
    lessons.set("lesson2", <Lesson2 />)
    root.render(lessons.get(this.state.name));
  }

  render() {
    return (
        <button onClick={() => this.loadLession() }>
          { this.state.name }
        </button>
      );      
 }
}

class LessonList extends React.Component {
  constructor(props) {
    console.log(props)
    super(props);
    this.state = {
    }
  }

  render() {
    return (
      <div>
        <Lesson lessonId='1' name='lesson1' description='scikit-learn: 数值回归预测'/>
        <br/>
        <br/>
        <Lesson lessonId='2' name='lesson2' description='pytorch: 手写数字识别'/>
      </div>
      );      
 }
}


const domContainer1 = document.querySelector('#lesson_list_container');
const root = ReactDOM.createRoot(domContainer1);
root.render(<LessonList />);
